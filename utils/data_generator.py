
# File 4: utils/data_generator.py - Creates training batches

"""
Data Generator for Training
Creates batches of MRI data for the model to learn from
"""

import numpy as np
import tensorflow as tf
import os
import glob
import nibabel as nib
from scipy.ndimage import rotate

class BraTSDataGenerator(tf.keras.utils.Sequence):
    """
    Data Generator for BraTS dataset
    
    What it does:
    - Loads MRI scans in batches (not all at once, saves memory)
    - Applies data augmentation (random flips, rotations)
    - Prepares data for training
    """
    
    def __init__(self, data_dir, patient_ids, batch_size=1, target_shape=(128, 128, 128), 
                 n_channels=4, n_classes=4, shuffle=True, augment=True):
        """
        Initialize generator
        
        data_dir: Folder containing all patient folders
        patient_ids: List of patient folder names to use
        batch_size: How many patients per batch (usually 1 for 3D due to memory)
        target_shape: Size to resize images to
        n_channels: Number of MRI modalities (4: T1, T1ce, T2, FLAIR)
        n_classes: Number of segmentation classes (4: bg, necrotic, edema, enhancing)
        shuffle: Whether to randomize order each epoch
        augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        
        self.indexes = np.arange(len(self.patient_ids))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.patient_ids) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch of patient indices
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_patient_ids = [self.patient_ids[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(batch_patient_ids)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def load_patient(self, patient_id):
        """Load all MRI scans for one patient"""
        patient_dir = os.path.join(self.data_dir, patient_id)
        
        # Try different file naming patterns
        modalities = ['t1', 't1ce', 't2', 'flair']
        patterns = [
            '{}_{}.nii.gz',
            '{}-{}.nii.gz',
            '*{}.nii.gz',
            '*_{}*.nii.gz'
        ]
        
        modality_data = []
        
        for mod in modalities:
            file_path = None
            for pattern in patterns:
                search_pattern = os.path.join(patient_dir, pattern.format(patient_id, mod))
                files = glob.glob(search_pattern)
                if files:
                    file_path = files[0]
                    break
            
            if file_path is None:
                raise FileNotFoundError(f"Could not find {mod} file for {patient_id}")
            
            # Load and preprocess
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            
            # Normalize
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                data = (data - mean) / std
            
            # Resize
            from scipy.ndimage import zoom
            factors = [t / c for t, c in zip(self.target_shape, data.shape)]
            data = zoom(data, factors, order=1)
            
            modality_data.append(data)
        
        # Stack into 4D array (H, W, D, 4)
        mri_volume = np.stack(modality_data, axis=-1)
        
        # Load segmentation mask
        seg_patterns = [
            '{}_seg.nii.gz',
            '{}-seg.nii.gz',
            '*_seg.nii.gz'
        ]
        
        seg_path = None
        for pattern in seg_patterns:
            search_pattern = os.path.join(patient_dir, pattern.format(patient_id))
            files = glob.glob(search_pattern)
            if files:
                seg_path = files[0]
                break
        
        if seg_path is None:
            raise FileNotFoundError(f"Could not find segmentation file for {patient_id}")
        
        seg_img = nib.load(seg_path)
        seg_mask = seg_img.get_fdata()
        
        # Resize segmentation (use nearest neighbor to preserve labels)
        factors = [t / c for t, c in zip(self.target_shape, seg_mask.shape)]
        seg_mask = zoom(seg_mask, factors, order=0)
        
        # Convert BraTS labels: 0,1,2,4 → 0,1,2,3
        seg_mask[seg_mask == 4] = 3
        seg_mask = seg_mask.astype(np.uint8)
        
        return mri_volume, seg_mask
    
    def augment_data(self, mri, mask):
        """Apply random augmentations"""
        if np.random.random() > 0.5:
            # Random flip along X axis
            mri = np.flip(mri, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        
        if np.random.random() > 0.7:
            # Random rotation (small angle)
            angle = np.random.uniform(-10, 10)
            axes = (0, 1)
            mri = rotate(mri, angle, axes=axes, reshape=False, order=1)
            mask = rotate(mask, angle, axes=axes, reshape=False, order=0)
        
        return mri, mask
    
    def __data_generation(self, batch_patient_ids):
        """Generate data for one batch"""
        X = np.zeros((len(batch_patient_ids), *self.target_shape, self.n_channels), dtype=np.float32)
        y = np.zeros((len(batch_patient_ids), *self.target_shape, self.n_classes), dtype=np.float32)
        
        for i, patient_id in enumerate(batch_patient_ids):
            try:
                mri_volume, seg_mask = self.load_patient(patient_id)
                
                # Augment
                if self.augment:
                    mri_volume, seg_mask = self.augment_data(mri_volume, seg_mask)
                
                # Convert mask to one-hot encoding
                seg_onehot = tf.keras.utils.to_categorical(seg_mask, num_classes=self.n_classes)
                
                X[i] = mri_volume
                y[i] = seg_onehot
                
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
        
        return X, y

def create_data_split(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split patients into train/val/test sets
    
    Returns: (train_ids, val_ids, test_ids)
    """
    # Get all patient folders
    patient_ids = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    patient_ids.sort()
    
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    
    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    print(f"Data split: {n_total} total patients")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  Val: {len(val_ids)} ({len(val_ids)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")
    
    return train_ids, val_ids, test_ids

# Example usage
if __name__ == "__main__":
    print("Data Generator module ready!")
    print("Use this to create training batches from your MRI dataset.")


with open('/mnt/kimi/output/brain_tumor_segmentation/utils/data_generator.py', 'w') as f:
    f.write(data_gen_code)

print("✅ Created: utils/data_generator.py")
print("\nThis file:")
print("  - Loads MRI data in batches (saves memory)")
print("  - Applies data augmentation (flips, rotations)")
print("  - Splits data into train/validation/test sets")
