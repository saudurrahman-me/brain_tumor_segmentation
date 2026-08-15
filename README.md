# Brain Tumor Segmentation

An advanced deep learning AI system for automatic brain tumor segmentation from 3D MRI scans.

This project utilizes a **3D U-Net Neural Network** to analyze multimodal MRI data and pinpoint different regions of a brain tumor, acting as a clinical assistant to measure and visualize tumor progression based on BraTS standards.

## 🧠 How It Works

The system takes raw NIfTI (`.nii.gz`) brain scans and processes them through a complete pipeline:

1. **Multimodal Input:** Processes 4 distinct MRI modalities simultaneously:
   - **T1** (Native)
   - **T1ce** (Post-contrast)
   - **T2**
   - **FLAIR** (Fluid-attenuated inversion recovery)
2. **Preprocessing:** Automatically normalizes image intensities (z-score) and resizes 3D volumes to a standard `128x128x128` grid.
3. **U-Net Architecture:** Features a contracting encoder to extract complex features and an expanding decoder with skip connections to accurately reconstruct the precise boundaries of the tumor.
4. **Multi-class Segmentation:** The AI classifies every voxel into 4 categories:
   - Background (Healthy tissue)
   - Necrotic Core (Dead tissue)
   - Edema (Swelling)
   - Enhancing Tumor (Active tumor)

## 📊 Clinical Analysis & Reporting

The system includes a complete `predict.py` module and a GUI that not only generates the segmentation masks but also provides real-world clinical metrics:
- Calculates the precise physical volume in cubic centimeters (`cc`) based on voxel spacing.
- Provides standard metrics: **Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)**.
- Generates automated JSON/Text clinical reports and visual overlays.

## 🛠️ Technology Stack

- **Deep Learning Framework:** TensorFlow / Keras
- **Medical Imaging:** NiBabel
- **Data Processing:** NumPy, SciPy
- **Visualization:** Matplotlib
- **Interface:** Custom GUI built in Python

## 🚀 Usage

### 1. Training the Model
```bash
python train.py
```
*Note: Ensure your dataset is correctly placed in the `data/` directory before initiating training.*

### 2. Running Predictions
To generate a segmentation mask and clinical report for a new patient:
```bash
python predict.py --model ./results/models/best_model.h5 --input ./patient_folder
```

### 3. Launching the GUI
You can use the interactive Graphical User Interface to visualize the predictions:
```bash
python demo.py
```

## 🗂️ Project Structure

```text
brain_tumor_segmentation/
│
├── models/             # U-Net 3D Architecture (unet_model.py)
├── utils/              # Data generators, MRI preprocessing, and evaluation scripts
├── gui/                # Interactive Graphical User Interface components
├── data/               # MRI dataset directory
├── train.py            # Main script to train the model
├── predict.py          # Inference script to generate masks and clinical reports
├── demo.py             # Entry point for the GUI application
└── requirements.txt    # Python package dependencies
```

## 📝 License
This project is available under the [MIT License](LICENSE).
