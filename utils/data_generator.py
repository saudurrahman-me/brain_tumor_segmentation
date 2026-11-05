"""Memory-conscious patient-wise data loading for BraTS-style NIfTI datasets."""
from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate

from utils.mri_preprocessing import MRIPreprocessor


class BraTSDataGenerator(tf.keras.utils.Sequence):
    """Load one or more patients per batch; failures are raised, never hidden."""

    def __init__(
        self,
        data_dir,
        patient_ids,
        batch_size=1,
        target_shape=(128, 128, 128),
        n_channels=4,
        n_classes=4,
        shuffle=True,
        augment=True,
        seed=42,
    ):
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.patient_ids = list(patient_ids)
        self.batch_size = int(batch_size)
        self.target_shape = tuple(int(v) for v in target_shape)
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.n_channels != 4 or self.n_classes != 4:
            raise ValueError("This BraTS pipeline expects four input modalities and four classes")
        if not self.patient_ids:
            raise ValueError("No patient IDs were supplied to the data generator")
        self.indexes = np.arange(len(self.patient_ids))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.patient_ids) / self.batch_size))

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.batch_size
        batch_indexes = self.indexes[start:start + self.batch_size]
        patient_ids = [self.patient_ids[i] for i in batch_indexes]
        return self._make_batch(patient_ids)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    @staticmethod
    def remap_brats_labels(mask):
        """Map original BraTS labels 0/1/2/4 to contiguous 0/1/2/3."""
        mask = np.rint(mask).astype(np.int16)
        labels = set(np.unique(mask).tolist())
        unsupported = labels - {0, 1, 2, 3, 4}
        if unsupported:
            raise ValueError(f"Unsupported segmentation labels: {sorted(unsupported)}")
        mask[mask == 4] = 3
        return mask.astype(np.uint8)

    def load_patient(self, patient_id):
        patient_dir = os.path.join(self.data_dir, patient_id)
        preprocessor = MRIPreprocessor(self.target_shape)
        patient_data = preprocessor.preprocess_patient_data(patient_dir)
        if "seg" not in patient_data:
            raise FileNotFoundError(
                f"No segmentation file was found for training patient {patient_id}"
            )
        images = preprocessor.create_multimodal_input(patient_data)
        mask = self.remap_brats_labels(patient_data["seg"])
        if mask.shape != self.target_shape:
            raise ValueError(
                f"Segmentation shape {mask.shape} does not match {self.target_shape} "
                f"for patient {patient_id}"
            )
        return images, mask

    def augment_data(self, mri, mask):
        if self.rng.random() < 0.5:
            mri = np.flip(mri, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if self.rng.random() < 0.3:
            angle = float(self.rng.uniform(-10.0, 10.0))
            mri = rotate(mri, angle, axes=(0, 1), reshape=False, order=1, mode="nearest")
            mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0, mode="nearest")
        return mri.astype(np.float32), mask.astype(np.uint8)

    def _make_batch(self, patient_ids):
        x = np.empty(
            (len(patient_ids), *self.target_shape, self.n_channels), dtype=np.float32
        )
        y = np.empty(
            (len(patient_ids), *self.target_shape, self.n_classes), dtype=np.float32
        )
        for index, patient_id in enumerate(patient_ids):
            try:
                mri, mask = self.load_patient(patient_id)
                if self.augment:
                    mri, mask = self.augment_data(mri, mask)
                x[index] = mri
                y[index] = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)
            except Exception as exc:
                raise RuntimeError(f"Failed to load patient '{patient_id}': {exc}") from exc
        return x, y


def create_data_split(data_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create deterministic patient-level train/validation/test splits."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1")
    patient_ids = sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith(".")
    )
    if len(patient_ids) < 3:
        raise ValueError(
            f"At least three patient folders are required for train/validation/test splits; "
            f"found {len(patient_ids)}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    n_total = len(patient_ids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_train = n_total - 2
        n_val = 1
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    print(
        f"Data split: {n_total} patients | train={len(train_ids)}, "
        f"validation={len(val_ids)}, test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids
