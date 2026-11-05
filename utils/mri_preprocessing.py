"""Load, align, normalize, and resize multimodal brain MRI volumes.

The output grid preserves the reference image's physical field of view and affine.
BraTS-style labels are retained here; remapping is handled by the data generator.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


class MRIPreprocessor:
    """Preprocess T1, T1ce, T2, FLAIR and optional segmentation NIfTI files."""

    MODALITY_ALIASES = {
        "t1": ("t1", "t1n"),
        "t1ce": ("t1ce", "t1c", "t1gd"),
        "t2": ("t2", "t2w"),
        "flair": ("flair", "t2f", "fla"),
        "seg": ("seg", "mask"),
    }
    REQUIRED_MODALITIES = ("t1", "t1ce", "t2", "flair")

    def __init__(self, target_shape: Sequence[int] = (128, 128, 128)):
        self.target_shape = tuple(int(v) for v in target_shape)
        if len(self.target_shape) != 3 or any(v <= 0 for v in self.target_shape):
            raise ValueError("target_shape must contain three positive integers")
        self.modalities = list(self.REQUIRED_MODALITIES)
        self.last_affine: Optional[np.ndarray] = None
        self.last_header = None
        self.voxel_spacing_mm: Optional[tuple] = None
        self.loaded_paths: Dict[str, str] = {}

    @staticmethod
    def _is_nifti(filename: str) -> bool:
        lower = filename.lower()
        return lower.endswith(".nii") or lower.endswith(".nii.gz")

    @classmethod
    def _matches_alias(cls, filename: str, aliases: Sequence[str]) -> bool:
        stem = filename.lower()
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        for alias in aliases:
            if re.search(r"(?:^|[_-])" + re.escape(alias) + r"(?:$|[_-])", stem):
                return True
        return False

    def find_patient_files(self, patient_dir: str) -> Dict[str, str]:
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory does not exist: {patient_dir}")
        files = sorted(
            os.path.join(patient_dir, name)
            for name in os.listdir(patient_dir)
            if self._is_nifti(name) and os.path.isfile(os.path.join(patient_dir, name))
        )
        found: Dict[str, str] = {}
        for modality, aliases in self.MODALITY_ALIASES.items():
            matches = [
                path for path in files
                if self._matches_alias(os.path.basename(path), aliases)
            ]
            if matches:
                if len(matches) > 1:
                    # Prefer the shortest, most explicit filename deterministically.
                    matches.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
                found[modality] = matches[0]
        missing = [name for name in self.REQUIRED_MODALITIES if name not in found]
        if missing:
            raise FileNotFoundError(
                f"Missing required MRI modality/mod modalities in {patient_dir}: "
                + ", ".join(missing)
            )
        self.loaded_paths = found
        return found

    @staticmethod
    def normalize_intensity(image: np.ndarray) -> np.ndarray:
        """Z-score non-zero voxels so background zeros remain zero."""
        image = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(image)
        foreground = finite & (image != 0)
        if not np.any(foreground):
            raise ValueError("MRI volume contains no finite non-zero voxels")
        values = image[foreground]
        mean = float(values.mean())
        std = float(values.std())
        output = np.zeros_like(image, dtype=np.float32)
        if std > 1e-8:
            output[foreground] = (values - mean) / std
        else:
            output[foreground] = values - mean
        return output

    def _target_grid(self, reference_img):
        """Build a resized grid with the same physical field of view."""
        source_shape = np.asarray(reference_img.shape[:3], dtype=np.float64)
        target_shape = np.asarray(self.target_shape, dtype=np.int32)
        scale = source_shape / target_shape
        voxel_transform = np.eye(4, dtype=np.float64)
        voxel_transform[:3, :3] = np.diag(scale)
        voxel_transform[:3, 3] = (scale - 1.0) / 2.0
        target_affine = reference_img.affine @ voxel_transform
        return tuple(int(v) for v in target_shape), target_affine

    def preprocess_patient_data(self, patient_dir: str) -> Dict[str, np.ndarray]:
        """Return aligned float32 modalities and an optional uint8 segmentation."""
        paths = self.find_patient_files(patient_dir)
        images = {
            name: nib.as_closest_canonical(nib.load(paths[name]))
            for name in self.REQUIRED_MODALITIES
        }
        reference = images["t1"]
        if len(reference.shape) != 3:
            raise ValueError(f"T1 image must be 3D, got shape {reference.shape}")
        target = self._target_grid(reference)
        data: Dict[str, np.ndarray] = {}

        for modality in self.REQUIRED_MODALITIES:
            image = images[modality]
            if len(image.shape) != 3:
                raise ValueError(f"{modality} image must be 3D, got shape {image.shape}")
            aligned = resample_from_to(image, target, order=1)
            data[modality] = self.normalize_intensity(aligned.get_fdata(dtype=np.float32))

        if "seg" in paths:
            seg_img = nib.as_closest_canonical(nib.load(paths["seg"]))
            if len(seg_img.shape) != 3:
                raise ValueError(f"Segmentation must be 3D, got shape {seg_img.shape}")
            aligned_seg = resample_from_to(seg_img, target, order=0)
            seg = np.rint(aligned_seg.get_fdata(dtype=np.float32)).astype(np.int16)
            labels = set(np.unique(seg).tolist())
            unsupported = labels - {0, 1, 2, 3, 4}
            if unsupported:
                raise ValueError(f"Unsupported segmentation labels: {sorted(unsupported)}")
            data["seg"] = seg.astype(np.uint8)

        self.last_affine = np.asarray(target[1], dtype=np.float64)
        self.last_header = reference.header.copy()
        self.last_header.set_data_shape(self.target_shape)
        self.last_header.set_data_dtype(np.uint8)
        self.voxel_spacing_mm = tuple(float(v) for v in nib.affines.voxel_sizes(self.last_affine))
        return data

    def create_multimodal_input(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        missing = [name for name in self.REQUIRED_MODALITIES if name not in data]
        if missing:
            raise ValueError("Missing MRI modalities: " + ", ".join(missing))
        arrays = [np.asarray(data[name], dtype=np.float32) for name in self.REQUIRED_MODALITIES]
        expected = self.target_shape
        for name, array in zip(self.REQUIRED_MODALITIES, arrays):
            if array.shape != expected:
                raise ValueError(f"{name} has shape {array.shape}; expected {expected}")
        return np.stack(arrays, axis=-1).astype(np.float32, copy=False)
