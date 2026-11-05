"""3D U-Net model and foreground-aware segmentation loss/metrics."""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


def foreground_dice(y_true, y_pred, smooth=1e-6):
    """Mean soft Dice over foreground classes, averaged across the batch."""
    y_true = tf.cast(y_true[..., 1:], tf.float32)
    y_pred = tf.cast(y_pred[..., 1:], tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    scores = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(scores)


def foreground_dice_loss(y_true, y_pred):
    return 1.0 - foreground_dice(y_true, y_pred)


def combined_segmentation_loss(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cross_entropy) + foreground_dice_loss(y_true, y_pred)


def foreground_mean_iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true[..., 1:], tf.float32)
    y_pred = tf.cast(y_pred[..., 1:], tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true + y_pred, axis=axes) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


class UNet3D:
    """Four-level 3D U-Net for four-channel MRI input and four output classes."""

    def __init__(self, input_shape=(128, 128, 128, 4), num_classes=4, base_filters=16):
        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.base_filters = int(base_filters)
        self.model = None
        if len(self.input_shape) != 4 or self.input_shape[-1] != 4:
            raise ValueError("input_shape must be (D, H, W, 4)")
        if self.num_classes != 4:
            raise ValueError("This implementation expects four segmentation classes")
        if self.base_filters < 4:
            raise ValueError("base_filters must be at least 4")
        if any(dim is None or dim % 16 != 0 for dim in self.input_shape[:3]):
            raise ValueError("All spatial input dimensions must be divisible by 16")

    @staticmethod
    def conv_block(inputs, filters, name):
        x = layers.Conv3D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(inputs)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        x = layers.Conv3D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        return layers.Activation("relu", name=f"{name}_relu2")(x)

    def encoder_block(self, inputs, filters, name):
        features = self.conv_block(inputs, filters, name)
        pooled = layers.MaxPooling3D(pool_size=2, name=f"{name}_pool")(features)
        pooled = layers.Dropout(0.2, name=f"{name}_dropout")(pooled)
        return features, pooled

    def decoder_block(self, inputs, skip_features, filters, name):
        x = layers.Conv3DTranspose(
            filters, kernel_size=2, strides=2, padding="same", name=f"{name}_upconv"
        )(inputs)
        x = layers.Concatenate(name=f"{name}_concat")([x, skip_features])
        return self.conv_block(x, filters, name)

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name="multimodal_mri")
        c1, p1 = self.encoder_block(inputs, self.base_filters, "encoder1")
        c2, p2 = self.encoder_block(p1, self.base_filters * 2, "encoder2")
        c3, p3 = self.encoder_block(p2, self.base_filters * 4, "encoder3")
        c4, p4 = self.encoder_block(p3, self.base_filters * 8, "encoder4")
        bridge = self.conv_block(p4, self.base_filters * 16, "bridge")
        d4 = self.decoder_block(bridge, c4, self.base_filters * 8, "decoder4")
        d3 = self.decoder_block(d4, c3, self.base_filters * 4, "decoder3")
        d2 = self.decoder_block(d3, c2, self.base_filters * 2, "decoder2")
        d1 = self.decoder_block(d2, c1, self.base_filters, "decoder1")
        outputs = layers.Conv3D(self.num_classes, 1, activation="softmax", name="class_probabilities")(d1)
        self.model = Model(inputs, outputs, name="UNet3D_BrainTumor")
        return self.model

    def compile_model(self, learning_rate=1e-4):
        if self.model is None:
            self.build_model()
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=combined_segmentation_loss,
            metrics=["categorical_accuracy", foreground_dice, foreground_mean_iou],
        )
        return self.model

    def get_model_summary(self):
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def save_model(self, filepath):
        if self.model is None:
            raise RuntimeError("Build or train the model before saving it")
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath, compile=False)
        self.input_shape = tuple(self.model.input_shape[1:])
        return self.model


if __name__ == "__main__":
    network = UNet3D()
    network.build_model()
    network.compile_model()
    network.get_model_summary()
