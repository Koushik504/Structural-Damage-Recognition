"""
EXPERIMENT 3: Center-Crop Region-Focused Model Training

Option C uses exactly one center crop per original training image.
The preprocessing output must have:
  X: (N, 224, 224, 3)
  y: (N, 2)

The original-image-level stratified train/validation split is retained.
The test set is never used here. The baseline model is not loaded or overwritten.
"""

import os
import csv
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

REGION_X_PATH = "results/gradcam_region/preprocessed/X_train_region.npy"
REGION_Y_PATH = "results/gradcam_region/preprocessed/y_train_region.npy"
ORIG_Y_PATH = "DATA/task2_damage_state_1/task2_y_train.npy"

OUTPUT_MODEL_PATH = "models/mobilenetv1_gradcam_region.keras"
HISTORY_PATH = "results/gradcam_region/mobilenetv1_gradcam_region_history.csv"
RESULTS_DIR = "results/gradcam_region"

VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PATIENCE = 5

# Option C: one crop per original image.
N_CROPS = 1


class RegionGenerator(tf.keras.utils.Sequence):
    """Yield one preprocessed image per original-image index."""

    def __init__(self, X_crops, y_crops, orig_indices,
                 n_crops=1, batch_size=32, shuffle=False):
        if n_crops != 1:
            raise ValueError("Option C requires n_crops=1.")
        self.X_crops = X_crops
        self.y_crops = y_crops
        self.indices = np.asarray(orig_indices, dtype=np.int64).copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        return (
            np.asarray(self.X_crops[batch_indices], dtype=np.float32),
            np.asarray(self.y_crops[batch_indices], dtype=np.float32),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for path in (REGION_X_PATH, REGION_Y_PATH, ORIG_Y_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading Option C center-crop arrays with memory mapping...")
    X_crops = np.load(REGION_X_PATH, mmap_mode="r")
    y_crops = np.load(REGION_Y_PATH, mmap_mode="r")
    y_orig = np.load(ORIG_Y_PATH)

    print(f"X shape: {X_crops.shape}, dtype={X_crops.dtype}")
    print(f"y shape: {y_crops.shape}, dtype={y_crops.dtype}")
    print(f"Original labels shape: {y_orig.shape}")

    if X_crops.ndim != 4 or X_crops.shape[1:] != (224, 224, 3):
        raise ValueError(
            f"Expected X shape (N,224,224,3), got {X_crops.shape}"
        )
    if y_crops.ndim != 2 or y_crops.shape[1] != 2:
        raise ValueError(f"Expected one-hot labels shape (N,2), got {y_crops.shape}")
    if y_orig.ndim != 2 or y_orig.shape[1] != 2:
        raise ValueError(f"Expected original one-hot labels shape (N,2), got {y_orig.shape}")
    if len(X_crops) != len(y_crops):
        raise ValueError("Preprocessed image and label counts do not match.")
    if len(X_crops) != len(y_orig):
        raise ValueError(
            "Option C requires exactly one preprocessed image per original image. "
            f"Got {len(X_crops)} crops and {len(y_orig)} original labels."
        )
    if not np.array_equal(np.asarray(y_crops), np.asarray(y_orig)):
        raise ValueError(
            "Preprocessed labels do not match original labels in order. "
            "Do not train until preprocessing output is checked."
        )

    print("\nReproducing original 90/10 stratified split...")
    labels = np.argmax(y_orig, axis=1)
    indices = np.arange(len(y_orig))

    train_orig_idx, val_orig_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Training images:   {len(train_orig_idx)}")
    print(f"Validation images: {len(val_orig_idx)}")
    print("Each original image maps to exactly one center crop.")

    train_gen = RegionGenerator(
        X_crops, y_crops, train_orig_idx,
        n_crops=N_CROPS, batch_size=BATCH_SIZE, shuffle=True,
    )
    val_gen = RegionGenerator(
        X_crops, y_crops, val_orig_idx,
        n_crops=N_CROPS, batch_size=BATCH_SIZE, shuffle=False,
    )

    print("\nBuilding fresh MobileNetV1 model...")
    base_model = tf.keras.applications.MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print(f"\n{'=' * 60}")
    print(f"Training Option C center-crop model (LR={LR:.0e}, max {EPOCHS} epochs)")
    print(f"Saving separately to: {OUTPUT_MODEL_PATH}")
    print(f"{'=' * 60}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        for i, (acc, loss, val_acc, val_loss) in enumerate(zip(
            history.history["accuracy"],
            history.history["loss"],
            history.history["val_accuracy"],
            history.history["val_loss"],
        ), start=1):
            writer.writerow([i, acc, loss, val_acc, val_loss])

    print(f"\nModel saved to   : {OUTPUT_MODEL_PATH}")
    print(f"History saved to : {HISTORY_PATH}")
    print(f"Training time    : {elapsed:.1f}s")
    print("\nRun evaluation next:")
    print(
        f"  python -m src.evaluate_experiment "
        f"--model {OUTPUT_MODEL_PATH} "
        f"--results {RESULTS_DIR}/mobilenetv1_gradcam_region_results.json"
    )


if __name__ == "__main__":
    main()
