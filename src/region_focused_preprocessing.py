"""
Experiment 3 preprocessing: one center crop per original training image.

Inputs:
  DATA/task2_damage_state_2/task2_X_train.npy
  DATA/task2_damage_state_1/task2_y_train.npy

Outputs:
  results/gradcam_region/preprocessed/X_train_region.npy
  results/gradcam_region/preprocessed/y_train_region.npy

This is Option C: one center crop per source image, not five crops.
It preserves the original image-level train/validation split logic because
each original image produces exactly one crop.

Expected output X size is about 7.1 GB for 11,811 float32 images of
shape 224x224x3. Ensure enough disk space before running.
"""

from pathlib import Path
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
X_PATH = ROOT / "DATA" / "task2_damage_state_2" / "task2_X_train.npy"
Y_PATH = ROOT / "DATA" / "task2_damage_state_1" / "task2_y_train.npy"
OUT_DIR = ROOT / "results" / "gradcam_region" / "preprocessed"
X_OUT = OUT_DIR / "X_train_region.npy"
Y_OUT = OUT_DIR / "y_train_region.npy"
LOG_OUT = OUT_DIR / "preprocessing_log.json"

CROP_SIZE = 160
OUTPUT_SIZE = 224


def main():
    for path in (X_PATH, Y_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    print(f"Loading image array metadata: {X_PATH}", flush=True)
    X = np.load(X_PATH, mmap_mode="r")
    print(f"Loading labels: {Y_PATH}", flush=True)
    y = np.load(Y_PATH, mmap_mode="r")

    if X.ndim != 4 or X.shape[1:] != (224, 224, 3):
        raise ValueError(f"Expected X shape (N,224,224,3), got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"Image/label count mismatch: {len(X)} vs {len(y)}")

    n = len(X)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    estimated_bytes = n * OUTPUT_SIZE * OUTPUT_SIZE * 3 * np.dtype(np.float32).itemsize
    print(f"Original images: {n:,}", flush=True)
    print("Crops per image: 1 (center crop)", flush=True)
    print(f"Output image array estimate: {estimated_bytes / (1024**3):.2f} GiB", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)

    # Use .npy memory-mapped files so the entire output array is not held in RAM.
    X_out = np.lib.format.open_memmap(
        X_OUT, mode="w+", dtype=np.float32,
        shape=(n, OUTPUT_SIZE, OUTPUT_SIZE, 3)
    )
    Y_out = np.lib.format.open_memmap(
        Y_OUT, mode="w+", dtype=y.dtype, shape=y.shape
    )

    start = time.time()
    offset = (224 - CROP_SIZE) // 2

    try:
        for i in range(n):
            image = np.asarray(X[i])
            crop = image[offset:offset + CROP_SIZE, offset:offset + CROP_SIZE, :]
            resized = cv2.resize(
                crop, (OUTPUT_SIZE, OUTPUT_SIZE),
                interpolation=cv2.INTER_LINEAR
            )
            X_out[i] = resized.astype(np.float32, copy=False)
            Y_out[i] = y[i]

            if (i + 1) % 500 == 0 or i + 1 == n:
                elapsed = time.time() - start
                print(f"Processed {i + 1:,}/{n:,} images ({elapsed:.1f}s)", flush=True)

        X_out.flush()
        Y_out.flush()
    finally:
        del X_out
        del Y_out

    log = {
        "strategy": "single_center_crop",
        "source_image_count": int(n),
        "crops_per_image": 1,
        "crop_size": CROP_SIZE,
        "output_size": OUTPUT_SIZE,
        "X_input": str(X_PATH.relative_to(ROOT)),
        "y_input": str(Y_PATH.relative_to(ROOT)),
        "X_output": str(X_OUT.relative_to(ROOT)),
        "y_output": str(Y_OUT.relative_to(ROOT)),
        "X_output_shape": [int(n), OUTPUT_SIZE, OUTPUT_SIZE, 3],
        "y_output_shape": list(y.shape),
        "X_output_dtype": "float32",
        "elapsed_seconds": round(time.time() - start, 2),
    }
    LOG_OUT.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\nPreprocessing complete.", flush=True)
    print(f"X output: {X_OUT}", flush=True)
    print(f"y output: {Y_OUT}", flush=True)
    print(f"Log: {LOG_OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
