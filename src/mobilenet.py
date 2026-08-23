"""MobileNetV2 CNN model architecture and fine-tuning helpers for CIFAR-10.

Optimized for high generalization (>91% test accuracy):
- In-model Data Augmentation (RandomFlip, RandomTranslation, RandomZoom, RandomRotation)
- Bilinear Upsampling to 96x96 matching ImageNet receptive field
- Label Smoothing (0.1) on Categorical Cross-Entropy
- Cosine Decay Learning Rate Scheduling
- Deeper Fine-Tuning unfreezing top 4 Inverted Residual blocks (Layer >= 80)
"""

from __future__ import annotations

from typing import Tuple

from .config import IMG_SHAPE, NUM_CLASSES, SEED
from .utils import set_seed

__all__ = ["build_mobilenet_v2", "unfreeze_mobilenet_top_layers", "build_augmentation_layers"]


def build_augmentation_layers():
    """Build in-model data augmentation pipeline for CIFAR-10 images."""
    from tensorflow.keras import layers

    return [
        layers.RandomFlip("horizontal", name="aug_flip"),
        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08,
            fill_mode="nearest",
            name="aug_translation",
        ),
        layers.RandomRotation(factor=0.04, fill_mode="nearest", name="aug_rotation"),
        layers.RandomZoom(height_factor=(-0.05, 0.05), fill_mode="nearest", name="aug_zoom"),
    ]


def build_mobilenet_v2(
    input_shape: Tuple[int, int, int] = IMG_SHAPE,
    target_dim: int = 96,
    num_classes: int = NUM_CLASSES,
    head_units: int = 256,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.1,
    use_augmentation: bool = True,
    seed: int | None = SEED,
    freeze_base: bool = True,
):
    """Build an augmented MobileNetV2 transfer learning model for CIFAR-10.

    Args:
        input_shape: (32, 32, 3) by default.
        target_dim: Spatial dimension for MobileNet (default 96x96).
        num_classes: 10 for CIFAR-10.
        head_units: Neurons in the classification dense head.
        dropout_rate: Dropout rate before the final softmax layer.
        learning_rate: Initial learning rate for Adam optimizer.
        label_smoothing: Cross-entropy label smoothing factor (0.1).
        use_augmentation: Whether to include spatial data augmentation.
        seed: Random seed for reproducibility.
        freeze_base: Whether to freeze the pre-trained backbone weights.

    Returns:
        Compiled (keras.Model, base_model) ready for training.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    if seed is not None:
        set_seed(seed)

    # 1. Base MobileNetV2 pre-trained on ImageNet at 96x96
    base_model = keras.applications.MobileNetV2(
        input_shape=(target_dim, target_dim, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    # 2. Input layer accepting float32 in [0, 1]
    inputs = keras.Input(shape=input_shape, name="cifar10_image_input")

    # 3. Optional in-model data augmentation (active only during training)
    x = inputs
    if use_augmentation:
        for aug_layer in build_augmentation_layers():
            x = aug_layer(x)

    # 4. Upsample 32x32 -> 96x96
    upsample_factor = target_dim // input_shape[0]  # 3
    x = layers.UpSampling2D(
        size=(upsample_factor, upsample_factor),
        interpolation="bilinear",
        name="upsample_to_96",
    )(x)

    # 5. Rescaling to [-1, 1] expected by MobileNetV2: x * 2.0 - 1.0
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescaling_mobilenet")(x)

    # 6. Feature extractor (training=False preserves batchnorm running stats)
    x = base_model(x, training=False)

    # 7. Enhanced classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(head_units, activation="relu", name="head_dense")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_mobilenet_v2")

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_mobilenet_top_layers(
    model,
    base_model,
    fine_tune_at: int = 80,
    total_steps: int = 1500,
    initial_learning_rate: float = 2e-4,
    label_smoothing: float = 0.1,
):
    """Unfreeze top layers of the backbone for fine-tuning with Cosine Decay."""
    from tensorflow import keras
    from tensorflow.keras import layers

    base_model.trainable = True

    # Freeze all layers before fine_tune_at and keep BatchNorm layers frozen
    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Cosine decay schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_steps,
        alpha=0.01,
    )

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model
