
# File 6: predict.py - Prediction script
#!/usr/bin/env python3
"""
Prediction Script for Brain Tumor Segmentation

Use this AFTER training to segment new MRI scans.

What it does:
1. Loads a trained model
2. Takes new MRI scans as input
3. Outputs segmentation masks showing tumor locations
4. Generates clinical reports with measurements

Usage:
    python predict.py --model ./results/models/best_model.h5 --input ./patient_folder --output ./predictions
"""

import os
import sys
import argparse
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from models.unet_model import UNet3D
from utils.mri_preprocessing import MRIPreprocessor

class TumorAnalyzer:
    """Analyzes segmentation results and generates reports"""
    
    def __init__(self, voxel_spacing=(1.0, 1.0, 1.0)):
        """
        voxel_spacing: Physical size of each voxel in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.voxel_volume_cc = np.prod(voxel_spacing) / 1000  # Convert to cm³
    
    def analyze_segmentation(self, seg_mask, patient_id="Unknown"):
        """
        Analyze segmentation mask and generate clinical report
        
        seg_mask: 3D array with labels (0=bg, 1=necrotic, 2=edema, 3=enhancing)
        """
        # Calculate volumes for each tumor region
        volumes = {}
        for label, name in [(1, 'Necrotic'), (2, 'Edema'), (3, 'Enhancing')]:
            pixel_count = np.sum(seg_mask == label)
            volume_cc = pixel_count * self.voxel_volume_cc
            volumes[name] = {
                'pixels': int(pixel_count),
                'volume_cc': float(volume_cc)
            }
        
        # Calculate combined regions (BraTS standards)
        wt_pixels = np.sum(seg_mask > 0)  # Whole Tumor (all regions)
        tc_pixels = np.sum((seg_mask == 1) | (seg_mask == 3))  # Tumor Core
        et_pixels = np.sum(seg_mask == 3)  # Enhancing Tumor only
        
        report = {
            'patient_id': patient_id,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'voxel_spacing_mm': self.voxel_spacing,
            'volumes': {
                'Whole_Tumor_cc': float(wt_pixels * self.voxel_volume_cc),
                'Tumor_Core_cc': float(tc_pixels * self.voxel_volume_cc),
                'Enhancing_Tumor_cc': float(et_pixels * self.voxel_volume_cc),
                'Necrotic_cc': volumes['Necrotic']['volume_cc'],
                'Edema_cc': volumes['Edema']['volume_cc']
            },
            'details': volumes
        }
        
        return report
    
    def visualize_segmentation(self, mri_volume, seg_mask, patient_id, save_dir):
        """Create visualization of segmentation results"""
        # Take middle slice for visualization
        slice_idx = mri_volume.shape[2] // 2
        
        mri_slice = mri_volume[:, :, slice_idx, 0]  # T1 modality
        seg_slice = seg_mask[:, :, slice_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original MRI
        axes[0].imshow(mri_slice, cmap='gray')
        axes[0].set_title(f'{patient_id}\\nOriginal MRI (T1)')
        axes[0].axis('off')
        
        # Segmentation mask (colored)
        seg_rgb = np.zeros((*seg_slice.shape, 3))
        seg_rgb[seg_slice == 1] = [1, 0, 0]  # Red = Necrotic
        seg_rgb[seg_slice == 2] = [0, 1, 0]  # Green = Edema
        seg_rgb[seg_slice == 3] = [1, 1, 0]  # Yellow = Enhancing
        axes[1].imshow(seg_rgb)
        axes[1].set_title('Segmentation\\nRed=Necrotic, Green=Edema, Yellow=Enhancing')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(mri_slice, cmap='gray', alpha=0.7)
        axes[2].imshow(seg_rgb, alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{patient_id}_segmentation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

def predict_single_patient(model, patient_dir, output_dir, preprocessor):
    """Run prediction on one patient"""
    patient_id = os.path.basename(patient_dir)
    print(f"\\n🔍 Processing: {patient_id}")
    
    # Load and preprocess MRI
    print("  Loading MRI scans...")
    patient_data = preprocessor.preprocess_patient_data(patient_dir)
    
    if not patient_data:
        print(f"  ❌ Error: Could not load data from {patient_dir}")
        return None
    
    # Create multimodal input
    mri_volume = preprocessor.create_multimodal_input(patient_data)
    print(f"  Input shape: {mri_volume.shape}")
    
    # Add batch dimension (model expects batch of patients)
    input_batch = np.expand_dims(mri_volume, axis=0)
    
    # Run prediction
    print("  Running segmentation...")
    prediction = model.predict(input_batch, verbose=0)
    
    # Convert to class labels (0, 1, 2, 3)
    seg_mask = np.argmax(prediction[0], axis=-1)
    print(f"  Output shape: {seg_mask.shape}")
    
    # Analyze results
    analyzer = TumorAnalyzer()
    report = analyzer.analyze_segmentation(seg_mask, patient_id)
    
    # Create output folder for this patient
    patient_output = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    # Save segmentation as NIfTI file
    seg_nifti = nib.Nifti1Image(seg_mask.astype(np.uint8), affine=np.eye(4))
    nib.save(seg_nifti, os.path.join(patient_output, 'segmentation.nii.gz'))
    
    # Save report as JSON
    with open(os.path.join(patient_output, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save report as text
    report_text = f"""
BRAIN TUMOR SEGMENTATION REPORT
{'='*50}
Patient ID: {patient_id}
Date: {report['analysis_date']}

TUMOR VOLUMES
{'='*50}
Whole Tumor:        {report['volumes']['Whole_Tumor_cc']:.2f} cc
Tumor Core:         {report['volumes']['Tumor_Core_cc']:.2f} cc
Enhancing Tumor:    {report['volumes']['Enhancing_Tumor_cc']:.2f} cc
Necrotic Core:      {report['volumes']['Necrotic_cc']:.2f} cc
Edema:              {report['volumes']['Edema_cc']:.2f} cc

{'='*50}
End of Report
"""
    with open(os.path.join(patient_output, 'report.txt'), 'w') as f:
        f.write(report_text)
    
    # Create visualization
    viz_path = analyzer.visualize_segmentation(mri_volume, seg_mask, patient_id, patient_output)
    
    # Print summary
    print(f"\\n  📊 Results:")
    print(f"     Whole Tumor: {report['volumes']['Whole_Tumor_cc']:.2f} cc")
    print(f"     Enhancing: {report['volumes']['Enhancing_Tumor_cc']:.2f} cc")
    print(f"  💾 Saved to: {patient_output}")
    
    return report

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict Brain Tumor Segmentation')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to patient folder or directory with multiple patients')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Where to save predictions (default: predictions)')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple patients (input should be directory)')
    
    args = parser.parse_args()
    
    print("\\n" + "="*60)
    print("BRAIN TUMOR SEGMENTATION - PREDICTION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Load model
    print("\\n📥 Loading model...")
    unet = UNet3D()
    unet.load_model(args.model)
    unet.compile_model()
    print("✅ Model loaded successfully")
    
    # Initialize preprocessor
    preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process patients
    if args.batch:
        # Multiple patients
        patient_dirs = [os.path.join(args.input, d) for d in os.listdir(args.input)
                       if os.path.isdir(os.path.join(args.input, d))]
        print(f"\\n🔄 Processing {len(patient_dirs)} patients...\\n")
        
        for i, patient_dir in enumerate(patient_dirs):
            print(f"[{i+1}/{len(patient_dirs)}] ", end="")
            try:
                predict_single_patient(unet.model, patient_dir, args.output, preprocessor)
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        # Single patient
        predict_single_patient(unet.model, args.input, args.output, preprocessor)
    
    print("\\n" + "="*60)
    print("✅ PREDICTION COMPLETE!")
    print("="*60)
    print(f"Results saved in: {args.output}/")

if __name__ == "__main__":
    main()


with open('/mnt/kimi/output/brain_tumor_segmentation/predict.py', 'w') as f:
    f.write(predict_code)

print("✅ Created: predict.py")
print("\nThis script uses trained model to segment new MRI scans.")
print("Run after training with: python predict.py --model ./results/models/best_model.h5 --input ./patient")
