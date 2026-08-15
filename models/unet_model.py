
# File 3: models/unet_model.py - The AI model architecture

"""
U-Net Model for Brain Tumor Segmentation
This is the neural network that will learn to identify tumors
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class UNet3D:
    """
    3D U-Net Architecture
    
    What is U-Net?
    - A neural network designed for medical image segmentation
    - Looks like a "U" shape: encoder (down) → bottleneck → decoder (up)
    - Encoder: Extracts features (learns what tumors look like)
    - Decoder: Reconstructs the segmentation mask
    - Skip connections: Help preserve fine details
    
    Input: 3D MRI scan (128 x 128 x 128 x 4)
           - 128x128x128 = spatial dimensions (height, width, depth)
           - 4 = 4 MRI modalities (T1, T1ce, T2, FLAIR)
    
    Output: 3D segmentation mask (128 x 128 x 128 x 4)
           - 4 classes: Background, Necrotic, Edema, Enhancing Tumor
    """
    
    def __init__(self, input_shape=(128, 128, 128, 4), num_classes=4, base_filters=32):
        """
        Initialize the model
        
        input_shape: Size of input MRI (H, W, D, channels)
        num_classes: Number of tumor types to identify (0=bg, 1=necrotic, 2=edema, 3=enhancing)
        base_filters: Number of filters in first layer (more = more complex model)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.model = None
    
    def conv_block(self, inputs, filters, name):
        """
        Convolution block: Two 3D convolutions with batch normalization
        
        This extracts features from the MRI at each level
        """
        # First convolution
        x = layers.Conv3D(filters, 3, padding='same', name=f'{name}_conv1')(inputs)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)
        
        # Second convolution
        x = layers.Conv3D(filters, 3, padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = layers.Activation('relu', name=f'{name}_relu2')(x)
        
        return x
    
    def encoder_block(self, inputs, filters, name):
        """
        Encoder block: Conv → MaxPool (downsampling)
        
        Reduces spatial size by half, doubles feature channels
        """
        x = self.conv_block(inputs, filters, name)
        p = layers.MaxPooling3D(2, name=f'{name}_pool')(x)
        p = layers.Dropout(0.3, name=f'{name}_dropout')(p)  # Prevent overfitting
        return x, p  # Return x for skip connection, p for next level
    
    def decoder_block(self, inputs, skip_features, filters, name):
        """
        Decoder block: Upsample → Concatenate with skip → Conv
        
        Reconstructs the segmentation mask
        """
        # Upsample by 2x
        x = layers.Conv3DTranspose(filters, 2, strides=2, padding='same',
                                   name=f'{name}_upconv')(inputs)
        # Add skip connection (preserves fine details)
        x = layers.concatenate([x, skip_features], name=f'{name}_concat')
        x = self.conv_block(x, filters, name)
        return x
    
    def build_model(self):
        """Build the complete U-Net architecture"""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ENCODER (Contracting Path) - extracts features
        # Level 1: 128³ → 64³
        c1, p1 = self.encoder_block(inputs, self.base_filters, 'encoder1')
        
        # Level 2: 64³ → 32³
        c2, p2 = self.encoder_block(p1, self.base_filters * 2, 'encoder2')
        
        # Level 3: 32³ → 16³
        c3, p3 = self.encoder_block(p2, self.base_filters * 4, 'encoder3')
        
        # Level 4: 16³ → 8³
        c4, p4 = self.encoder_block(p3, self.base_filters * 8, 'encoder4')
        
        # BOTTLENECK - deepest level
        bridge = self.conv_block(p4, self.base_filters * 16, 'bridge')
        
        # DECODER (Expanding Path) - reconstructs mask
        # Level 4: 8³ → 16³
        d4 = self.decoder_block(bridge, c4, self.base_filters * 8, 'decoder4')
        
        # Level 3: 16³ → 32³
        d3 = self.decoder_block(d4, c3, self.base_filters * 4, 'decoder3')
        
        # Level 2: 32³ → 64³
        d2 = self.decoder_block(d3, c2, self.base_filters * 2, 'decoder2')
        
        # Level 1: 64³ → 128³
        d1 = self.decoder_block(d2, c1, self.base_filters, 'decoder1')
        
        # OUTPUT - final segmentation
        outputs = layers.Conv3D(self.num_classes, 1, activation='softmax',
                               name='output')(d1)
        
        self.model = Model(inputs, outputs, name='UNet3D_BrainTumor')
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile the model with optimizer and loss function
        """
        if self.model is None:
            self.build_model()
        
        # Custom metrics
        def dice_coefficient(y_true, y_pred, smooth=1.0):
            """Dice score: measures overlap between prediction and truth (0-1, higher is better)"""
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        
        def iou_metric(y_true, y_pred, smooth=1.0):
            """IoU (Intersection over Union): another overlap measure"""
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
            return (intersection + smooth) / (union + smooth)
        
        # Combined loss: Dice loss + Crossentropy
        def combined_loss(y_true, y_pred):
            cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            dice_loss = 1 - dice_coefficient(y_true, y_pred)
            return cce + dice_loss
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[
                'accuracy',
                dice_coefficient,
                iou_metric
            ]
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model architecture"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath, compile=False)
        print(f"Model loaded from: {filepath}")

# Example usage
if __name__ == "__main__":
    # Create model
    unet = UNet3D(input_shape=(128, 128, 128, 4), num_classes=4)
    model = unet.build_model()
    unet.compile_model()
    
    print("\\n" + "="*60)
    print("3D U-Net Model Created Successfully!")
    print("="*60)
    unet.get_model_summary()


with open('/mnt/kimi/output/brain_tumor_segmentation/models/unet_model.py', 'w') as f:
    f.write(unet_code)

print("✅ Created: models/unet_model.py")
print("\nThis is the AI model (3D U-Net) that:")
print("  - Takes 4 MRI scans as input (128x128x128x4)")
print("  - Outputs segmentation mask (128x128x128x4)")
print("  - Identifies 4 classes: background + 3 tumor types")
