# Running the project

## 1. Prepare the environment

Use Python 3.10 or 3.11, create a virtual environment, and install requirements.txt. Select a TensorFlow wheel compatible with your operating system and hardware.

## 2. Prepare the dataset

Create one folder per patient. Each training folder must contain T1, T1ce, T2, FLAIR and a segmentation NIfTI volume. Confirm the dataset's label specification: this code maps BraTS-style labels 0/1/2/4 to 0/1/2/3.

## 3. Train

    python train.py --data_dir ./data --output_dir ./results --epochs 50 --batch_size 1

Training writes the patient split, best and final model checkpoints, CSV history, training plot, and held-out test metrics under results/.

## 4. Predict

    python predict.py --model ./results/models/best_model.keras --input ./data/Patient_001

The output directory contains a NIfTI segmentation, JSON volume report, and PNG overlay.

## 5. Launch the desktop GUI

    python demo.py

Load the trained .keras model, load a patient folder, run prediction, and save results.

## Research checks

- Confirm modality filenames and segmentation labels for the dataset.
- Inspect alignment and orientation of several preprocessed patients.
- Verify that patient IDs do not overlap between train, validation, and test splits.
- Evaluate on the held-out test set after model selection; do not tune against it.
- Report measured per-patient and aggregate metrics only. Do not invent or extrapolate results.
- This software is not a clinically validated diagnostic system.
