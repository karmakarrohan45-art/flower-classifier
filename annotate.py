
#!/usr/bin/env python3
"""
annotate.py — Basic label inspection and correction tool

Displays images from the tf_flowers dataset alongside their auto-labelled
classes and lets you correct any mislabelled ones.  Corrections are saved
to corrected_labels.csv.

Usage:
    python annotate.py            # annotate 50 random images
    python annotate.py --limit 20 # annotate only 20 images
"""

import os
import sys
import csv
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

TFDS_DIR = "dataset"
CSV_PATH = "corrected_labels.csv"
IMAGE_SIZE = 180
NUM_CLASSES = 5
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def load_images(num_images=50):
    ds = tfds.load(
        "tf_flowers",
        split="train",
        as_supervised=True,
        shuffle_files=True,
        data_dir=TFDS_DIR,
        download=True,
    )
    ds = ds.shuffle(1000).take(num_images)
    results = []
    for image, label in ds:
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.uint8).numpy()
        results.append((image, int(label.numpy())))
    return results


def display_image(image, filename):
    pil = Image.fromarray(image)
    pil.save(filename)
    return filename


def annotate(num_images):
    print(f"Annotating {num_images} images from tf_flowers ...")
    print("For each image:\n"
          "  ENTER = keep label,  0-4 = correct,  s = skip\n")

    images = load_images(num_images)
    corrections = []

    for idx, (image, true_idx) in enumerate(images):
        true_name = CLASS_NAMES[true_idx]
        fname = display_image(image, f"temp_{idx}.png")
        print(f"\n[{idx + 1}/{num_images}]  {fname}")
        print(f"  Label: {true_name} (index {true_idx})")
        print(f"  Classes: {list(enumerate(CLASS_NAMES))}")

        raw = input("  New label [ENTER/skip/0-4]: ").strip().lower()

        if raw == "s":
            print("  Skipped")
        elif raw == "":
            print(f"  Kept: {true_name}")
        else:
            try:
                new_idx = int(raw)
                if 0 <= new_idx < NUM_CLASSES:
                    new_name = CLASS_NAMES[new_idx]
                    print(f"  Corrected: {true_name} -> {new_name}")
                    corrections.append({
                        "image_index": idx,
                        "old_label": true_name,
                        "new_label": new_name,
                        "note": fname,
                    })
                else:
                    print("  Out of range, keeping original")
            except ValueError:
                print("  Invalid input, keeping original")

        if os.path.exists(fname):
            os.remove(fname)

    if corrections:
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_index", "old_label", "new_label", "note"])
            if write_header:
                writer.writeheader()
            for row in corrections:
                writer.writerow(row)
        print(f"\n{len(corrections)} corrections saved -> {CSV_PATH}")
    else:
        print("\nNo corrections made.")

    for i in range(num_images):
        fname = f"temp_{i}.png"
        if os.path.exists(fname):
            os.remove(fname)


if __name__ == "__main__":
    limit = 50
    if len(sys.argv) > 2 and sys.argv[1] == "--limit":
        limit = int(sys.argv[2])
    annotate(limit)
