"""
================================================================================
EXPERIMENT 1: MobileNetV1 Fine-Tuning
================================================================================

Hypothesis:
    Unfreezing the final convolutional blocks of the pretrained MobileNetV1
    backbone and training them end-to-end with a very small learning rate will
    allow the network to adapt low-level feature detectors to structural-damage
    imagery, improving over the frozen-backbone 76.78% baseline.

Methodology:
    Single fine-tuning phase:
        Load the saved baseline model (weights already trained with frozen backbone).
        Freeze all layers first, then selectively unfreeze from UNFREEZE_FROM_LAYER
        onward (conv_dw_12 and all subsequent layers — 16 layers total across blocks
        12 and 13 plus the classification head).
        Use a drastically reduced learning rate (1e-5) to avoid catastrophic
        forgetting of ImageNet features in the earlier layers.
        Train until EarlyStopping triggers on val_loss.

    NOTE: The saved model (mobilenetv1.keras) is a *flat* Keras functional model —
    all MobileNetV1 layers are direct members of model.layers (indices 0-89).
    model.layers[0] is an InputLayer; there is no nested sub-model.  Layer
    traversal must therefore iterate model.layers, not backbone.layers.

Variables Changed vs Baseline:
    - Backbone trainability: frozen -> partially unfrozen (last block group)
    - Learning rate: 1e-4 (head) -> 1e-5 (fine-tune phase)
    - Starting weights: random head + frozen backbone -> trained head + backbone

Variables Kept Constant:
    - Dataset and split (random_state=42, 90/10 stratified)
    - Image size: 224x224x3
    - Classification head architecture: GAP -> Dense(256) -> Dropout(0.5) -> Softmax(2)
    - Batch size: 32
    - Loss: categorical_crossentropy
    - EarlyStopping + ModelCheckpoint on val_loss

Expected Effect:
    +1% to +3% accuracy improvement. Fine-tuning too many layers or with too
    large a LR risks catastrophic forgetting and accuracy regression.

Evaluation Metrics:
    Accuracy, Precision, Recall, F1, Confusion Matrix, Training time
================================================================================
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

# --- Paths ---
BASELINE_MODEL_PATH = "models/mobilenetv1.keras"
OUTPUT_MODEL_PATH   = "models/mobilenetv1_finetuned.keras"
HISTORY_PATH        = "results/training/mobilenetv1_finetuned_history.csv"
RESULTS_DIR         = "results/training"

# --- Data ---
TRAIN_X_PATH = "DATA/task2_damage_state_2/task2_X_train.npy"
TRAIN_Y_PATH = "DATA/task2_damage_state_1/task2_y_train.npy"

# --- Split ---
VALIDATION_SIZE = 0.1
RANDOM_STATE    = 42

# --- Training hyper-parameters ---
BATCH_SIZE = 32

# Fine-tune phase (last conv blocks unfrozen, tiny LR)
FINETUNE_EPOCHS   = 15    # max; EarlyStopping will cut this short
FINETUNE_LR       = 1e-5
FINETUNE_PATIENCE = 4     # EarlyStopping patience

# --- Which MobileNetV1 layers to unfreeze for fine-tuning ---
# MobileNetV1 (tf.keras.applications.MobileNet) depthwise separable blocks:
#   conv_dw_1  ... conv_dw_13  (depthwise convolutions)
#   conv_pw_1  ... conv_pw_13  (pointwise convolutions)
# We unfreeze only the last block group (blocks 12 and 13) to minimise
# catastrophic forgetting while still adapting to the target domain.
UNFREEZE_FROM_LAYER = "conv_dw_12"   # unfreeze this layer and all after it


# ============================================================
# HELPERS
# ============================================================

def load_and_split():
    """Load training arrays and produce a reproducible stratified split."""
    print("Loading training data (memory-mapped)...")
    X = np.load(TRAIN_X_PATH, mmap_mode="r")
    y = np.load(TRAIN_Y_PATH)

    print(f"Full training set: {X.shape}  labels: {y.shape}")

    labels = np.argmax(y, axis=1)
    indices = np.arange(len(y))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train samples : {len(train_idx)}")
    print(f"Val   samples : {len(val_idx)}")
    return X, y, train_idx, val_idx


class IndexedGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient Keras Sequence that loads images by index from a
    memory-mapped NumPy array.  Optionally shuffles each epoch.
    """

    def __init__(self, X, y, indices, batch_size=32, shuffle=False):
        self.X          = X
        self.y          = y
        self.indices    = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self._current   = self.indices.copy()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self._current) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self._current))
        batch = self._current[start:end]
        return (
            np.asarray(self.X[batch], dtype=np.float32),
            np.asarray(self.y[batch], dtype=np.float32),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._current)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ---- 1. Load data ----
    X, y, train_idx, val_idx = load_and_split()

    train_gen = IndexedGenerator(X, y, train_idx, BATCH_SIZE, shuffle=True)
    val_gen   = IndexedGenerator(X, y, val_idx,   BATCH_SIZE, shuffle=False)

    # ---- 2. Load the baseline model ----
    print(f"\nLoading baseline model from: {BASELINE_MODEL_PATH}")
    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    model.summary()

    # ---- 3. Freeze all layers, then unfreeze from UNFREEZE_FROM_LAYER onward ----
    #
    # The saved model is a *flat* Keras functional model: all MobileNetV1 layers
    # are direct top-level entries in model.layers (indices 0-89).  There is no
    # nested sub-model — model.layers[0] is an InputLayer, not a MobileNet object.
    # We therefore iterate model.layers directly.
    #
    for layer in model.layers:
        layer.trainable = False

    reached = False
    unfrozen_names = []
    for layer in model.layers:
        if layer.name == UNFREEZE_FROM_LAYER:
            reached = True
        if reached:
            layer.trainable = True
            unfrozen_names.append(layer.name)

    print(f"\nUnfrozen layers ({len(unfrozen_names)}): {unfrozen_names}")

    # ---- 4. Fine-tune last conv blocks ----
    print(f"\n{'='*60}")
    print(f"FINE-TUNING from '{UNFREEZE_FROM_LAYER}' (LR={FINETUNE_LR:.0e})")
    print(f"{'='*60}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=FINETUNE_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    t0 = time.time()
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINETUNE_EPOCHS,
        callbacks=finetune_callbacks,
        verbose=1,
    )
    finetune_time = time.time() - t0
    print(f"\nFine-tuning complete in {finetune_time:.1f}s")

    # ---- 5. Save training history ----
    print(f"\nSaving history to: {HISTORY_PATH}")

    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        for i, (acc, loss, val_acc, val_loss) in enumerate(zip(
            history_ft.history["accuracy"],
            history_ft.history["loss"],
            history_ft.history["val_accuracy"],
            history_ft.history["val_loss"],
        )):
            writer.writerow([i, acc, loss, val_acc, val_loss])

    print(f"\nFine-tuned model saved to : {OUTPUT_MODEL_PATH}")
    print(f"Training history saved to : {HISTORY_PATH}")
    print(f"\nTotal fine-tuning time    : {finetune_time:.1f}s")
    print("\nRun the evaluation script next:")
    print(f"  python -m src.evaluate_experiment "
          f"--model {OUTPUT_MODEL_PATH} "
          f"--results {RESULTS_DIR}/mobilenetv1_finetuned_results.json")


if __name__ == "__main__":
    main()
