
# File 5: train.py - Main training script
#!/usr/bin/env python3
"""
Training Script for Brain Tumor Segmentation

This is the MAIN script you run to train the AI model.

What it does:
1. Loads your MRI dataset
2. Creates training and validation batches
3. Builds the 3D U-Net model
4. Trains the model for specified epochs
5. Saves the best model
6. Plots training history

Usage:
    python train.py --data_dir ./data --output_dir ./results --epochs 50
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import our modules
from models.unet_model import UNet3D
from utils.data_generator import BraTSDataGenerator, create_data_split

def setup_gpu():
    """Configure GPU memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"⚠️ GPU error: {e}")
    else:
        print("ℹ️ No GPU found, using CPU (will be slower)")

def train_model(data_dir, output_dir, epochs=50, batch_size=1, learning_rate=0.0001):
    """
    Main training function
    
    Parameters:
    - data_dir: Path to your dataset folder
    - output_dir: Where to save results
    - epochs: How many times to go through the entire dataset
    - batch_size: How many patients per batch (keep at 1 for 3D)
    - learning_rate: How fast the model learns (0.0001 is good default)
    """
    
    print("\\n" + "="*60)
    print("BRAIN TUMOR SEGMENTATION - TRAINING")
    print("="*60)
    
    # Setup
    setup_gpu()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    # Step 1: Split data into train/val/test
    print("\\n📊 Step 1: Splitting data...")
    train_ids, val_ids, test_ids = create_data_split(data_dir)
    
    # Save split info
    with open(f"{output_dir}/data_split.json", 'w') as f:
        json.dump({'train': train_ids, 'val': val_ids, 'test': test_ids}, f, indent=2)
    
    # Step 2: Create data generators
    print("\\n🔄 Step 2: Creating data generators...")
    
    train_gen = BraTSDataGenerator(
        data_dir=data_dir,
        patient_ids=train_ids,
        batch_size=batch_size,
        target_shape=(128, 128, 128),
        shuffle=True,
        augment=True  # Random flips/rotations for training
    )
    
    val_gen = BraTSDataGenerator(
        data_dir=data_dir,
        patient_ids=val_ids,
        batch_size=batch_size,
        target_shape=(128, 128, 128),
        shuffle=False,
        augment=False  # No augmentation for validation
    )
    
    print(f"   Training batches: {len(train_gen)}")
    print(f"   Validation batches: {len(val_gen)}")
    
    # Step 3: Build model
    print("\\n🏗️ Step 3: Building 3D U-Net model...")
    unet = UNet3D(
        input_shape=(128, 128, 128, 4),  # H, W, D, 4 modalities
        num_classes=4,  # bg, necrotic, edema, enhancing
        base_filters=32
    )
    
    model = unet.build_model()
    unet.compile_model(learning_rate=learning_rate)
    
    print(f"   Model parameters: {model.count_params():,}")
    
    # Step 4: Setup callbacks (save best model, stop early if not improving)
    print("\\n⚙️ Step 4: Setting up training callbacks...")
    
    callbacks = [
        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{output_dir}/models/best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Stop training if validation loss doesn't improve for 10 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate if stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Log training history
        tf.keras.callbacks.CSVLogger(f"{output_dir}/training_log.csv")
    ]
    
    # Step 5: Train!
    print("\\n🚀 Step 5: Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   This may take a while...\\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Save final model
    final_model_path = f"{output_dir}/models/final_model.h5"
    model.save(final_model_path)
    print(f"\\n💾 Final model saved: {final_model_path}")
    
    # Step 7: Plot training history
    print("\\n📈 Step 6: Plotting training history...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/training_history.png"
    plt.savefig(plot_path, dpi=150)
    print(f"   Plot saved: {plot_path}")
    
    print("\\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\\nResults saved in: {output_dir}/")
    print(f"Best model: {output_dir}/models/best_model.h5")
    print(f"\\nNext steps:")
    print(f"  1. Run prediction: python predict.py --model {output_dir}/models/best_model.h5 --input <patient_folder>")
    print(f"  2. Or launch GUI: python gui/segmentation_gui.py")
    
    return model, history

def main():
    """Parse command line arguments and start training"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Segmentation Model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset folder containing patient subfolders')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Where to save results (default: results)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size, keep at 1 for 3D (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    
    args = parser.parse_args()
    
    # Start training
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()


with open('/mnt/kimi/output/brain_tumor_segmentation/train.py', 'w') as f:
    f.write(train_code)

print("✅ Created: train.py")
print("\nThis is the MAIN training script.")
print("Run it with: python train.py --data_dir ./data --output_dir ./results")
