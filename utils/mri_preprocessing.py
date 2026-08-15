
# File 2: utils/mri_preprocessing.py - Handles loading and preparing MRI scans
# preprocessing_code =
"""
MRI Data Preprocessing Module
This file handles loading MRI scans and preparing them for the AI model
"""

import numpy as np
import nibabel as nib  # For reading NIfTI MRI files (.nii.gz)
import os
import glob
from scipy import ndimage

class MRIPreprocessor:
    """
    This class loads and preprocesses MRI brain scans
    
    What it does:
    1. Loads 4 types of MRI scans (T1, T1ce, T2, FLAIR)
    2. Normalizes the image brightness
    3. Resizes images to standard size
    4. Combines all 4 scans into one array
    """
    
    def __init__(self, target_shape=(128, 128, 128)):
        """
        Initialize the preprocessor
        
        target_shape: The size we want all images to be (default 128x128x128)
        """
        self.target_shape = target_shape
        self.modalities = ['t1', 't1ce', 't2', 'flair']
    
    def load_nifti_file(self, file_path):
        """
        Load a single MRI file (.nii.gz format)
        
        Returns: 3D numpy array of the brain scan
        """
        try:
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def normalize_intensity(self, image, method='z-score'):
        """
        Normalize image brightness to standard range
        
        z-score: Makes mean=0, std=1 (good for AI models)
        """
        if method == 'z-score':
            mean = np.mean(image)
            std = np.std(image)
            if std != 0:
                return (image - mean) / std
            return image - mean
        return image
    
    def resize_volume(self, volume, target_shape=None, order=1):
        """
        Resize 3D volume to target size
        
        order=1: Linear interpolation (smooth)
        order=0: Nearest neighbor (for masks, preserves labels)
        """
        if target_shape is None:
            target_shape = self.target_shape
        
        if volume.shape == target_shape:
            return volume
        
        # Calculate zoom factors for each dimension
        factors = [t / c for t, c in zip(target_shape, volume.shape)]
        resized = ndimage.zoom(volume, factors, order=order)
        return resized
    
    def preprocess_patient_data(self, patient_dir):
        """
        Load all 4 MRI scans for one patient
        
        patient_dir: Folder containing the MRI files
        Returns: Dictionary with all 4 modalities
        """
        data = {}
        
        # File patterns to look for (different datasets use different naming)
        file_patterns = {
            't1': ['*_t1.nii*', '*_t1n.nii*'],
            't1ce': ['*_t1ce.nii*', '*_t1c.nii*'],
            't2': ['*_t2.nii*', '*_t2w.nii*'],
            'flair': ['*_flair.nii*', '*_t2f.nii*', '*_fla.nii*'],
            'seg': ['*_seg.nii*']
        }
        
        for modality, patterns in file_patterns.items():
            # Try each pattern until we find a file
            for pattern in patterns:
                search_path = os.path.join(patient_dir, pattern)
                files = glob.glob(search_path)
                
                if files:
                    # Found a file, load it
                    volume = self.load_nifti_file(files[0])
                    if volume is not None:
                        # Normalize and resize
                        volume = self.normalize_intensity(volume)
                        volume = self.resize_volume(volume, self.target_shape)
                        data[modality] = volume
                        break
        
        return data
    
    def create_multimodal_input(self, data):
        """
        Stack all 4 MRI types into one array
        
        Input: Dictionary with t1, t1ce, t2, flair
        Output: Array of shape (H, W, D, 4) - 4 channels for 4 MRI types
        """
        modalities_list = []
        
        for mod in ['t1', 't1ce', 't2', 'flair']:
            if mod in data:
                modalities_list.append(data[mod])
        
        if len(modalities_list) == 0:
            raise ValueError("No MRI files found!")
        
        # Stack along the last dimension (channels)
        multimodal = np.stack(modalities_list, axis=-1)
        return multimodal

# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
    
    print("MRI Preprocessor ready!")
    print(f"Target shape: {preprocessor.target_shape}")
    print(f"Modalities: {preprocessor.modalities}")


with open('/mnt/kimi/output/brain_tumor_segmentation/utils/mri_preprocessing.py', 'w') as f:
    f.write(preprocessing_code)

print("✅ Created: utils/mri_preprocessing.py")
print("\nThis file handles:")
print("  - Loading .nii.gz MRI files")
print("  - Normalizing image brightness")
print("  - Resizing to 128x128x128")
print("  - Combining 4 MRI types into one array")
