#!/usr/bin/env python3
"""
evaluate_and_convert.py - Evaluate the trained model and convert to TF.js

This script:
  1. Loads the saved Keras model
  2. Evaluates it on the combined test set (tf_flowers + CIFAR-10)
  3. Converts the model to TensorFlow.js format for the React frontend

Usage:
    python evaluate_and_convert.py
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
NUM_CLASSES = 6


def preprocess(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_test_data():
    """Load tf_flowers + synthetic not_flower test set."""
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

    test_neg = create_synthetic_not_flower(643)
    print(f"  Synthetic (not_flower) - Test: 643")

    test_ds = test_f.concatenate(test_neg).shuffle(1000)
    labels = flower_labels + ["not_flower"]
    return test_ds, labels
    return test_ds, labels


def main():
    print("Loading test data (tf_flowers + synthetic) ...")
    test_ds, labels = load_test_data()
    print(f"Classes ({len(labels)}): {labels}")

    test_ds = (
        test_ds
        .map(lambda img, lbl: preprocess(img, lbl), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    model_path = os.path.join(MODEL_DIR, "flower_classifier.h5")
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    y_true, y_pred = [], []
    for images, lbls in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(lbls.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%")

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
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

    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report saved -> {report_path}")

    print(f"\nConverting model to TF.js format ...")

    from tensorflowjs.converters import converter

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

    shutil.copy(
        os.path.join(MODEL_DIR, "labels.json"),
        os.path.join(tfjs_dir, "labels.json"),
    )
    print(f"Labels copied -> {os.path.join(tfjs_dir, 'labels.json')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
