#!/usr/bin/env python3
"""
train_model.py - Flower Image Classification with Transfer Learning

This script trains a flower classifier:
  1. Loads the tf_flowers dataset (5 flower classes).
  2. Preprocesses and splits images into train/validation/test sets.
  3. Builds a MobileNetV2-based transfer learning model.
  4. Trains the model (stage 1: frozen base, stage 2: fine-tuning).
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

NUM_CLASSES = 5
IMAGE_SIZE = 180
BATCH_SIZE = 32
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE
MODEL_DIR = "model"
TFDS_DIR = "dataset"


def load_dataset():
    """Download (if needed) and load tf_flowers split into train/val/test."""
    print("Loading tf_flowers dataset ...")
    (train_ds, val_ds, test_ds), ds_info = tfds.load(
        "tf_flowers",
        split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        data_dir=TFDS_DIR,
        download=True,
    )
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val:   {len(val_ds)} images")
    print(f"  Test:  {len(test_ds)} images")
    print(f"  Classes: {ds_info.features['label'].names}")
    return train_ds, val_ds, test_ds, ds_info


def preprocess(image, label):
    """Resize + normalise image to [0, 1]."""
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    """Apply random flips, rotation, and color jittering for robustness."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Random 90-degree rotations (0, 1, 2, or 3 = 0°, 90°, 180°, 270°)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)
    # Clip to valid range after augmentation
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def prepare_dataset(ds, batch_size=BATCH_SIZE, shuffle=False, do_augment=True):
    """Map preprocessing, cache, augment, shuffle, batch, and prefetch."""
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
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="flower_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def fine_tune_model(model):
    """Unfreeze top layers of base model and recompile with lower LR."""
    base_model = model.layers[1]  # Rescaling is layer 0, base model is layer 1
    base_model.trainable = True

    # Freeze the earliest layers, fine-tune the last 20%
    total_layers = len(base_model.layers)
    trainable_start = int(total_layers * 0.8)
    for layer in base_model.layers[:trainable_start]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    print(f"Fine-tuning from layer {trainable_start} of {total_layers}")
    return model


def evaluate_model(model, test_ds, labels):
    """Compute accuracy, confusion matrix, and classification report."""
    y_true, y_pred = [], []
    for images, lbls in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(lbls.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%")

    report = classification_report(y_true, y_pred, target_names=labels)
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
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
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
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

    train_ds, val_ds, test_ds, ds_info = load_dataset()

    labels = ds_info.features["label"].names
    print(f"\nFlower classes: {labels}")

    train_ds = prepare_dataset(train_ds, shuffle=True, do_augment=True)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds)

    model = build_model()

    print("\n=== Stage 1: Training head (base frozen) ===")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )

    print("\n=== Stage 2: Fine-tuning (base trainable) ===")
    model = fine_tune_model(model)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )

    evaluate_model(model, test_ds, labels)
    save_model(model, labels)
    convert_to_tfjs()

    print("\nDone! Model is ready for the React frontend.")


if __name__ == "__main__":
    main()
