# Project summary

This repository contains a research implementation of multimodal 3D brain MRI segmentation using a 3D U-Net.

## Intended workflow

1. Discover T1, T1ce, T2, and FLAIR NIfTI volumes and align them in physical space.
2. Normalize non-zero intensities and resample to a target 3D grid.
3. Train a four-class 3D U-Net using patient-level train, validation, and test splits.
4. Evaluate WT, TC, ET, and foreground-class overlap on the held-out test set.
5. Export segmentation masks, physical-volume summaries, and visual overlays.

## Validation status

The branch contains code changes intended to improve pipeline correctness. These changes are not evidence of model performance. The updated pipeline has not yet been trained or validated against the project's actual dataset. Add measured test results only after running the experiment and checking data quality.

## Intended use

Academic experimentation and software development only. Not for clinical diagnosis or treatment decisions.
