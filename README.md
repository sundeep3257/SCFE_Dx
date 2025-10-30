## SCFE Detection Flask App

This app lets you upload a pelvis radiograph in NIfTI format (.nii or .nii.gz) or PNG (.png). If a PNG is uploaded, it is converted to NIfTI before analysis. The app runs a trained pipeline to crop around the femoral head, classifies for SCFE, and displays a composite image of the cropped input alongside an attention heatmap. It reports:

- Diagnosis: SCFE or No SCFE
- Probability of SCFE and probability of No SCFE
- Composite visualization image (cropped input and overlaid heatmap)

The app loads model weights from the `models` folder.

### Requirements

- Python 3.9+ recommended
- The model weight files must exist in `models/` with these names:
  - `models/Best_Large_Fem_Head_RN50.pth`
  - `models/SCFE_Classifier.pth`
- A CPU-only setup is supported. If you have a CUDA-capable GPU and drivers installed, the app will use it automatically.

### 1) Clone or download the repository

Download or clone this repository to your computer. Ensure the `models` folder with the two `.pth` files is present at the project root.

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install PyTorch

Install PyTorch first by following the official instructions for your system and Python version: `https://pytorch.org/get-started/locally/`

Examples:

CPU-only (Windows/macOS/Linux):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA CUDA 12.1 example (if you have a compatible GPU/driver):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If unsure, use the command recommended on the PyTorch website for your system.

### 4) Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 5) Run the app

```bash
python app.py
```

You should see output indicating the server is running on `http://127.0.0.1:5000`.

### 6) Use the app

1. Open your browser to `http://127.0.0.1:5000`.
2. Click the file picker and select a `.nii`, `.nii.gz`, or `.png` pelvis radiograph.
3. Click Analyze.
4. The result page will show the diagnosis, probabilities, and a composite image with the cropped input and the attention heatmap.

### File handling

- Uploads are placed in the `uploads/` folder.
- Generated visualization images are saved in `static/results/`.
- Temporary files are not retained beyond what is needed to display results.

### Troubleshooting

- If dependency installation fails, ensure PyTorch is installed first as described above, then run `pip install -r requirements.txt` again.
- If you see an error related to missing model files, verify both `.pth` files are present in the `models` folder and named exactly as listed.
- For `.nii.gz` uploads, ensure the file extension ends with `.nii.gz`.

### Notes

- The app automatically uses GPU if available; otherwise, it runs on CPU.
- Supported file types: `.nii`, `.nii.gz`, `.png`.

### Image requirements

- Upload images that contain a single hip joint only.
- If you are using a standard AP pelvis radiograph that shows both hips, crop the image down the midline into two separate images before uploading so that each image contains exactly one hip.


