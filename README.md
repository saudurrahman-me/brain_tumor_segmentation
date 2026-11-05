# Brain Tumor Segmentation

Research-oriented multimodal 3D MRI segmentation using a four-channel 3D U-Net.

**Research disclaimer:** This is an educational/research codebase, not a clinically validated diagnostic system. Do not use its outputs for diagnosis, treatment planning, or patient care.

## Pipeline

- Inputs: T1, T1ce, T2, and FLAIR NIfTI volumes (.nii or .nii.gz).
- Preprocessing: canonical orientation, alignment to the T1 reference grid, resampling to a configurable 3D shape, and non-zero-voxel z-score normalization.
- Network: four-level 3D U-Net with skip connections and four output classes.
- Labels: background 0, necrotic/non-enhancing core 1, edema 2, enhancing tumour 3. BraTS source label 4 is remapped to 3.
- Evaluation: patient-wise Dice and IoU for WT, TC, ET, and individual foreground classes.
- Outputs: NIfTI segmentation, JSON volume report, and central-slice visualization. Output NIfTI retains the resampled reference affine.

## Installation

Python 3.10 or 3.11 is recommended. Install a TensorFlow wheel compatible with your operating system and hardware.

    python -m venv .venv
    # macOS/Linux: source .venv/bin/activate
    # Windows PowerShell: .venv\\Scripts\\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Dataset structure

Use one folder per patient. Training requires all four modalities and a segmentation mask.

    data/
      Patient_001/
        Patient_001_t1.nii.gz
        Patient_001_t1ce.nii.gz
        Patient_001_t2.nii.gz
        Patient_001_flair.nii.gz
        Patient_001_seg.nii.gz

The loader recognizes common suffixes such as t1n, t1c, t2w, and t2f. Confirm filenames and label conventions against the documentation for your dataset.

## Train

    python train.py --data_dir ./data --output_dir ./results --epochs 50 --batch_size 1

The script creates patient-level train/validation/test splits (70/15/15), records them in results/data_split.json, selects the checkpoint using validation loss, and evaluates that checkpoint on the held-out test split. Test metrics are saved in results/test_metrics.json.

For lower memory use:

    python train.py --data_dir ./data --base_filters 8 --batch_size 1

## Predict

    python predict.py --model ./results/models/best_model.keras --input ./data/Patient_001 --output ./predictions

For a directory containing patient folders:

    python predict.py --model ./results/models/best_model.keras --input ./new_patients --batch --output ./predictions

Volumes are calculated using voxel spacing derived from the output NIfTI affine. They describe the predicted mask, not a diagnosis.

## GUI

    python demo.py

A graphical desktop session is required. Load a trained .keras model, load a patient folder, run prediction, and save results.

## Reproducibility and limitations

- A fixed seed is used for patient splitting and the default training run.
- Do not use the held-out test set for model selection or repeated hyperparameter tuning.
- Verify registration, skull stripping, segmentation label conventions, and patient-level split leakage for your dataset.
- Resampling to a fixed voxel grid changes voxel spacing; the affine is retained so physical volume calculations use the output grid spacing.
- This code has not yet been run against the actual project dataset in this environment. Inspect preprocessed patients and run tests before reporting results.
- Do not claim model accuracy or clinical performance without measured validation results.

## License

No license is currently declared. Add one only after confirming the project code may be redistributed.
