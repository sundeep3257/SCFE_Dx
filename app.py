import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import segmentation_models_pytorch as smp
import timm
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, send_file
import io
import zipfile

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

MODEL_INPUT_SIZE = (224, 224)
CROP_SIZE_HW = (244, 244)
CLASSIFIER_INPUT_SIZE_HW = (240, 240)
SCFE_CLASS_INDEX = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segmentation_model(weights_path, device):
    model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

class GuidedEfficientNetB1(nn.Module):
    def __init__(self, num_classes=2):
        super(GuidedEfficientNetB1, self).__init__()
        self.model = timm.create_model('efficientnet_b1', pretrained=False, in_chans=1, num_classes=num_classes)
        self.feature_extractor = self.model.conv_head
    def forward(self, x):
        features = self.model.forward_features(x)
        output = self.model.classifier(self.model.global_pool(features))
        return output, features

def predict_mask_3d(model, nifti_path, device, original_slice_shape_hw):
    transform_to_model_input_size_fn = T.Resize(MODEL_INPUT_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
    nifti_img = nib.load(nifti_path)
    img_data = nifti_img.get_fdata().astype(np.float32)
    if img_data.ndim == 2:
        num_slices = 1
        img_data = img_data[:, :, np.newaxis]
    elif img_data.ndim == 3:
        num_slices = img_data.shape[2]
    else:
        raise ValueError("Unsupported image dimensions")
    original_h, original_w = original_slice_shape_hw
    predicted_slices_resized_to_original = []
    for slice_idx in range(num_slices):
        img_slice_raw = img_data[:, :, slice_idx]
        min_val_slice = np.min(img_slice_raw)
        max_val_slice = np.max(img_slice_raw)
        if max_val_slice - min_val_slice > 1e-8:
            img_slice_normalized = (img_slice_raw - min_val_slice) / (max_val_slice - min_val_slice)
        else:
            img_slice_normalized = np.zeros_like(img_slice_raw)
        img_slice_tensor = torch.tensor(img_slice_normalized, dtype=torch.float32).unsqueeze(0)
        img_slice_transformed = transform_to_model_input_size_fn(img_slice_tensor)
        input_tensor = img_slice_transformed.unsqueeze(0).to(device)
        with torch.no_grad():
            output_logits = model(input_tensor)
        probs = torch.sigmoid(output_logits)
        predicted_mask_model_size = (probs > 0.5).squeeze().cpu()
        predicted_mask_model_size_unsqueezed = predicted_mask_model_size.unsqueeze(0).float()
        resize_back_transform = T.Resize((original_h, original_w), interpolation=InterpolationMode.NEAREST)
        original_size_mask_slice = resize_back_transform(predicted_mask_model_size_unsqueezed)
        original_size_mask_slice_np = original_size_mask_slice.squeeze().cpu().numpy().astype(np.uint8)
        predicted_slices_resized_to_original.append(original_size_mask_slice_np)
    final_3d_mask = np.stack(predicted_slices_resized_to_original, axis=2)
    return final_3d_mask, nifti_img.affine, nifti_img.header

def crop_by_mask_center(original_img_path, mask_data_2d):
    crop_height, crop_width = CROP_SIZE_HW
    seg_data = mask_data_2d
    roi_coords = np.argwhere(seg_data > 0)
    if roi_coords.size == 0:
        return None
    center_y, center_x = np.mean(roi_coords, axis=0).astype(int)
    half_h = crop_height // 2
    half_w = crop_width // 2
    y_min = max(center_y - half_h, 0)
    y_max = y_min + crop_height
    x_min = max(center_x - half_w, 0)
    x_max = x_min + crop_width
    orig_img = nib.load(original_img_path)
    orig_data = orig_img.get_fdata()
    if orig_data.ndim == 3 and orig_data.shape[-1] == 1:
        orig_data = np.squeeze(orig_data, axis=-1)
    if orig_data.ndim != 2:
        if orig_data.ndim > 2:
            orig_data = orig_data[:, :, 0]
    y_max = min(y_max, orig_data.shape[0])
    x_max = min(x_max, orig_data.shape[1])
    cropped = orig_data[y_min:y_max, x_min:x_max]
    if cropped.shape != (crop_height, crop_width):
        padded = np.zeros((crop_height, crop_width))
        y_offset = (crop_height - cropped.shape[0]) // 2
        x_offset = (crop_width - cropped.shape[1]) // 2
        padded[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped
        cropped = padded
    return cropped.astype(np.float32)

def preprocess_for_classifier(image_2d, resize_shape=(240, 240)):
    data = image_2d.astype(np.float32)
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    image_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(image_tensor, size=resize_shape, mode='bilinear', align_corners=False)
    return resized_tensor, data

def generate_composite_image(original_img_2d, heatmap_2d, out_path, title_text):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img_2d, cmap='gray')
    axes[0].set_title('Cropped Input')
    axes[0].axis('off')
    axes[1].imshow(original_img_2d, cmap='gray')
    axes[1].imshow(heatmap_2d, cmap='jet', alpha=0.5)
    axes[1].set_title(title_text)
    axes[1].axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

segmentation_weights_path = os.path.join(MODELS_FOLDER, 'Best_Large_Fem_Head_RN50.pth')
classifier_weights_path = os.path.join(MODELS_FOLDER, 'SCFE_Classifier.pth')
segmentation_model = load_segmentation_model(segmentation_weights_path, device)
classifier_model = GuidedEfficientNetB1(num_classes=2)
classifier_model.load_state_dict(torch.load(classifier_weights_path, map_location=device))
classifier_model.to(device)
classifier_model.eval()

def run_pipeline(nifti_path):
    temp_nifti_img = nib.load(nifti_path)
    original_dims_hw = (temp_nifti_img.shape[0], temp_nifti_img.shape[1])
    mask_data_3d, _, _ = predict_mask_3d(segmentation_model, nifti_path, device, original_dims_hw)
    if mask_data_3d.ndim == 3 and mask_data_3d.shape[-1] == 1:
        mask_data_2d = np.squeeze(mask_data_3d, axis=-1)
    elif mask_data_3d.ndim == 3:
        mask_data_2d = np.max(mask_data_3d, axis=2)
    else:
        mask_data_2d = mask_data_3d
    cropped_image = crop_by_mask_center(nifti_path, mask_data_2d)
    if cropped_image is None:
        return None
    input_tensor, original_cropped_image = preprocess_for_classifier(cropped_image, resize_shape=CLASSIFIER_INPUT_SIZE_HW)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs, feature_maps = classifier_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    scfe_prob = probabilities[0, SCFE_CLASS_INDEX].item()
    noscfe_prob = probabilities[0, 1 - SCFE_CLASS_INDEX].item()
    label = 'SCFE' if predicted_class == SCFE_CLASS_INDEX else 'No SCFE'
    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap_np, (original_cropped_image.shape[1], original_cropped_image.shape[0]))
    return {
        'label': label,
        'scfe_prob': float(scfe_prob),
        'noscfe_prob': float(noscfe_prob),
        'cropped_image': original_cropped_image,
        'heatmap': heatmap_resized
    }

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.nii', '.nii.gz', '.png'))

def png_to_nifti(png_path, out_dir, out_basename):
    img = Image.open(png_path)
    if img.mode != 'L':
        img = img.convert('L')
    img_array = np.array(img)
    nii_array = np.expand_dims(np.transpose(img_array, (1, 0)), axis=-1)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(nii_array, affine)
    out_name = f"{out_basename}.nii"
    out_path = os.path.join(out_dir, out_name)
    nib.save(nii_img, out_path)
    return out_name, out_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            lower = file.filename.lower()
            if lower.endswith('.nii'):
                filename = f"{file_id}.nii"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                stored_name = filename
            elif lower.endswith('.nii.gz'):
                filename = f"{file_id}.nii.gz"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                stored_name = filename
            elif lower.endswith('.png'):
                temp_png = os.path.join(UPLOAD_FOLDER, f"{file_id}.png")
                file.save(temp_png)
                nii_name, _ = png_to_nifti(temp_png, UPLOAD_FOLDER, file_id)
                stored_name = nii_name
            else:
                flash('Unsupported file type. Please upload a .nii, .nii.gz, or .png file')
                return redirect(request.url)
            return redirect(url_for('analyze', file_id=file_id, stored_name=stored_name))
        else:
            flash('Unsupported file type. Please upload a .nii, .nii.gz, or .png file')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    file_id = request.args.get('file_id')
    stored_name = request.args.get('stored_name')
    if not file_id or not stored_name:
        return redirect(url_for('index'))
    nifti_path = os.path.join(UPLOAD_FOLDER, stored_name)
    result = run_pipeline(nifti_path)
    if result is None:
        flash('No region of interest found. Please try another image.')
        return redirect(url_for('index'))
    out_image_name = f"{file_id}_composite.png"
    out_image_path = os.path.join(RESULTS_FOLDER, out_image_name)
    title_text = f"Attention Heatmap (Prediction: {result['label']})"
    generate_composite_image(result['cropped_image'], result['heatmap'], out_image_path, title_text)
    return render_template(
        'result.html',
        diagnosis=result['label'],
        scfe_prob=result['scfe_prob'],
        noscfe_prob=result['noscfe_prob'],
        composite_image=url_for('static', filename=f"results/{out_image_name}")
    )

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/download-test-images')
def download_test_images():
    static_dir = os.path.join(BASE_DIR, 'static')
    png_files = []
    for name in os.listdir(static_dir):
        path = os.path.join(static_dir, name)
        if os.path.isfile(path) and name.lower().endswith('.png'):
            png_files.append((name, path))
    if not png_files:
        flash('No test images found to download.')
        return redirect(url_for('index'))
    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, path in png_files:
            zf.write(path, arcname=name)
    mem_file.seek(0)
    return send_file(mem_file, mimetype='application/zip', as_attachment=True, download_name='test_images.zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

#