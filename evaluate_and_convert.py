#!/usr/bin/env python3
"""
evaluate_and_convert.py — Evaluate the trained model and convert to TF.js

This script:
  1. Loads the saved Keras model
  2. Evaluates it on the test split (accuracy, confusion matrix, report)
  3. Converts the model to TensorFlow.js format for the React frontend

Usage:
    python evaluate_and_convert.py
"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = "model"
TFDS_DIR = "dataset"
IMAGE_SIZE = 180
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def preprocess(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def main():
    print("Loading tf_flowers dataset ...")
    (_, _, test_ds), ds_info = tfds.load(
        "tf_flowers",
        split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        data_dir=TFDS_DIR,
        download=True,
    )

    labels = ds_info.features["label"].names
    print(f"Classes: {labels}")

    test_ds = (
        test_ds
        .map(lambda img, lbl: preprocess(img, lbl), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # Load model
    model_path = os.path.join(MODEL_DIR, "flower_classifier.h5")
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Evaluate
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
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {cm_path}")

    # Save classification report
    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report saved -> {report_path}")

    print(f"\nConverting model to TF.js format ...")

    from tensorflowjs.converters import converter

    # Use SavedModel format (fixes Keras 3 incompatibility with H5 conversion)
    saved_model_dir = os.path.join(MODEL_DIR, "saved_model")
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32)
    ])
    def serving_fn(x):
        return {"predictions": model(x)}

    tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": serving_fn})

    tfjs_dir = os.path.join("frontend", "public", "model")
    os.makedirs(tfjs_dir, exist_ok=True)

    args = [
        "--input_format", "tf_saved_model",
        "--output_format", "tfjs_graph_model",
        saved_model_dir,
        tfjs_dir,
    ]
    converter.convert(args)
    print(f"TF.js model saved -> {tfjs_dir}")

    # Copy label map
    shutil.copy(
        os.path.join(MODEL_DIR, "labels.json"),
        os.path.join(tfjs_dir, "labels.json"),
    )
    print(f"Labels copied -> {os.path.join(tfjs_dir, 'labels.json')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
