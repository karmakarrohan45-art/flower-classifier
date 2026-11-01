#!/usr/bin/env python3
"""
train_model.py - Flower Image Classification with Transfer Learning

This script trains a flower classifier that:
  1. Loads tf_flowers (5 flower classes) + CIFAR-10 (non-flower negatives).
  2. Preprocesses and splits images into train/validation/test sets.
  3. Builds a MobileNetV2-based transfer learning model (6 classes).
  4. Trains the model with data augmentation (base frozen).
  5. Evaluates it (accuracy, confusion matrix, classification report).
  6. Saves the trained model in Keras format.
  7. Converts the model to TensorFlow.js format for the React frontend.

Usage:
    python train_model.py
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NUM_CLASSES = 6
IMAGE_SIZE = 180
BATCH_SIZE = 64
EPOCHS_HEAD = 25
EPOCHS_FINE = 25
FINE_TUNE_LR = 5e-5
AUTOTUNE = tf.data.AUTOTUNE
MODEL_DIR = "model"
TFDS_DIR = "dataset"
EARLY_STOP_PATIENCE = 12
LABEL_SMOOTHING = 0.05


def load_dataset():
    """Load tf_flowers (5 classes) + synthetic not_flower images."""
    print("Loading tf_flowers dataset ...")
    (train_f, val_f, test_f), ds_info = tfds.load(
        "tf_flowers",
        split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        data_dir=TFDS_DIR,
        download=True,
    )
    flower_labels = list(ds_info.features["label"].names)
    print(f"  Flowers - Train: {len(train_f)}, Val: {len(val_f)}, Test: {len(test_f)}")
    print(f"  Classes: {flower_labels}")

    print("Generating synthetic not_flower images ...")

    def create_synthetic_not_flower(num_samples):
        def gen():
            import random
            for i in range(num_samples):
                pattern = random.randint(0, 3)
                if pattern == 0:
                    img = tf.random.uniform([IMAGE_SIZE, IMAGE_SIZE, 3], 0, 255, dtype=tf.float32)
                elif pattern == 1:
                    img = tf.random.normal([IMAGE_SIZE, IMAGE_SIZE, 3], 128, 60, dtype=tf.float32)
                    img = tf.clip_by_value(img, 0, 255)
                elif pattern == 2:
                    color = tf.random.uniform([], 0, 255)
                    img = tf.fill([IMAGE_SIZE, IMAGE_SIZE, 3], color)
                else:
                    row = tf.linspace(tf.random.uniform([], 0, 255), tf.random.uniform([], 0, 255), IMAGE_SIZE)
                    col = tf.linspace(tf.random.uniform([], 0, 255), tf.random.uniform([], 0, 255), IMAGE_SIZE)
                    grad = tf.tile(tf.expand_dims(row, 1), [1, IMAGE_SIZE])
                    img = tf.stack([grad, grad, grad], axis=-1)
                img = tf.cast(img, tf.uint8)
                yield img, tf.constant(NUM_CLASSES - 1, dtype=tf.int64)
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            )
        )

    train_neg = create_synthetic_not_flower(3000)
    val_neg = create_synthetic_not_flower(643)
    test_neg = create_synthetic_not_flower(643)
    print(f"  Synthetic (not_flower) - Train: 3000, Val: 643, Test: 643")

    train_ds = train_f.concatenate(train_neg).shuffle(4000)
    val_ds = val_f.concatenate(val_neg).shuffle(1000)
    test_ds = test_f.concatenate(test_neg).shuffle(1000)

    print(f"  Combined - Train: {len(train_f) + 3000}, Val: {len(val_f) + 643}, Test: {len(test_f) + 643}")
    class_names = flower_labels + ["not_flower"]
    print(f"  All classes: {class_names}")
    return train_ds, val_ds, test_ds, class_names


def preprocess(image, label):
    """Resize + normalise image to [0, 1]."""
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    """Apply random flips, rotation, brightness, saturation, contrast, hue, and cutout."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)
    
    # Random zoom/crop
    if tf.random.uniform([]) < 0.5:
        image = tf.image.central_crop(image, tf.random.uniform([], 0.8, 1.0))
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    
    # Random cutout (simplified)
    if tf.random.uniform([]) < 0.3:
        cutout_size = tf.random.uniform([], 16, 48, dtype=tf.int32)
        y = tf.random.uniform([], 0, IMAGE_SIZE - cutout_size, dtype=tf.int32)
        x = tf.random.uniform([], 0, IMAGE_SIZE - cutout_size, dtype=tf.int32)
        mask = tf.ones([cutout_size, cutout_size, 3], dtype=tf.float32)
        image = tf.tensor_scatter_nd_update(
            image,
            tf.stack(tf.meshgrid(tf.range(y, y + cutout_size), tf.range(x, x + cutout_size), indexing='ij'), axis=-1),
            tf.zeros([cutout_size, cutout_size, 3], dtype=tf.float32)
        )
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def prepare_dataset(ds, batch_size=BATCH_SIZE, shuffle=False, do_augment=True):
    """Map preprocessing, augment, shuffle, batch, and prefetch."""
    ds = ds.map(lambda img, lbl: preprocess(img, lbl), num_parallel_calls=AUTOTUNE)
    if do_augment:
        ds = ds.map(lambda img, lbl: augment(img, lbl), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE):
    """Transfer learning model with MobileNetV2 base."""
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    model = keras.Sequential([
        keras.Input(shape=(image_size, image_size, 3)),
        layers.Rescaling(scale=2.0, offset=-1.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.1),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax"),
    ], name="flower_classifier")

    # Cosine decay learning rate schedule
    total_steps = EPOCHS_HEAD * 200  # approximate steps per epoch
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        alpha=0.01
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    model.summary()
    return model, base_model


def tta_predict(model, images, tta_steps=8):
    """Test-time augmentation: average predictions over multiple augmentations."""
    preds_list = []
    for _ in range(tta_steps):
        aug_images = []
        for img in images:
            # Apply random augmentations
            aug = tf.image.random_flip_left_right(img)
            aug = tf.image.random_flip_up_down(aug)
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            aug = tf.image.rot90(aug, k=k)
            aug = tf.image.random_brightness(aug, 0.1)
            aug = tf.image.random_saturation(aug, 0.8, 1.2)
            aug = tf.image.random_contrast(aug, 0.8, 1.2)
            aug = tf.image.random_hue(aug, 0.02)
            aug = tf.clip_by_value(aug, 0.0, 1.0)
            aug_images.append(aug)
        aug_batch = tf.stack(aug_images)
        preds = model.predict(aug_batch, verbose=0)
        preds_list.append(preds)
    # Average predictions
    return np.mean(preds_list, axis=0)


def evaluate_model(model, test_ds, labels):
    """Compute accuracy, confusion matrix, and classification report with TTA."""
    y_true, y_pred = [], []
    for images, lbls in test_ds:
        # Use test-time augmentation for better accuracy
        preds = tta_predict(model, images, tta_steps=8)
        y_true.extend(lbls.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy (with TTA): {acc * 100:.2f}%")

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {out_path}")

    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy (with TTA): {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report saved -> {report_path}")
    return acc, report


def save_model(model, labels):
    """Save as H5 + JSON label map."""
    h5_path = os.path.join(MODEL_DIR, "flower_classifier.h5")
    model.save(h5_path)
    print(f"H5 model saved -> {h5_path}")

    label_map = {"labels": labels}
    label_path = os.path.join(MODEL_DIR, "labels.json")
    with open(label_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Labels saved -> {label_path}")


def convert_to_tfjs():
    """Convert the Keras model to TF.js format inside frontend/public/model."""
    from tensorflowjs.converters import converter
    import shutil

    h5_path = os.path.join(MODEL_DIR, "flower_classifier.h5")
    tfjs_dir = os.path.join("frontend", "public", "model")
    os.makedirs(tfjs_dir, exist_ok=True)

    model = tf.keras.models.load_model(h5_path)

    saved_model_dir = os.path.join(MODEL_DIR, "saved_model")
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32)])
    def serving_fn(x):
        return {"predictions": model(x)}

    tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": serving_fn})

    args = [
        "--input_format", "tf_saved_model",
        "--output_format", "tfjs_graph_model",
        saved_model_dir,
        tfjs_dir,
    ]
    converter.convert(args)
    print(f"TF.js model saved -> {tfjs_dir}")

    shutil.copy(
        os.path.join(MODEL_DIR, "labels.json"),
        os.path.join(tfjs_dir, "labels.json"),
    )
    print(f"Labels copied -> {os.path.join(tfjs_dir, 'labels.json')}")




def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_ds, val_ds, test_ds, labels = load_dataset()

    print(f"\nClasses ({len(labels)}): {labels}")

    train_ds = prepare_dataset(train_ds, shuffle=True, do_augment=True)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds)

    model, base_model = build_model()

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")

    def make_callbacks():
        return [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

    # Phase 1: Train the classifier head with base model frozen
    print("\n=== Phase 1: Training model (base frozen) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        verbose=1,
        callbacks=make_callbacks(),
    )

    # Phase 2: Fine-tune the base model with cosine decay LR
    print("\n=== Phase 2: Fine-tuning model (base unfrozen) ===")
    base_model.trainable = True
    
    fine_tune_steps = EPOCHS_FINE * 200
    fine_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FINE_TUNE_LR,
        decay_steps=fine_tune_steps,
        alpha=0.01
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_lr_schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        verbose=1,
        callbacks=make_callbacks(),
    )

    # Load the best model (lowest val_loss across both phases)
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"Loaded best model -> {checkpoint_path}")

    evaluate_model(model, test_ds, labels)
    save_model(model, labels)
    convert_to_tfjs()

    print("\nDone! Model is ready for the React frontend.")


if __name__ == "__main__":
    main()
