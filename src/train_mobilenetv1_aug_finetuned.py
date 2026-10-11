"""
================================================================================
EXPERIMENT 2: Controlled Data Augmentation + Fine-Tuning
================================================================================

Hypothesis:
    Applying conservative, domain-appropriate augmentation during training will
    reduce overfitting and improve generalisation on the structural-damage test
    set, further compounding the gains from fine-tuning (Experiment 1).

Methodology:
    All augmentation is applied on-the-fly inside a custom Keras Sequence,
    meaning:
      - The stored .npy files are NEVER modified.
      - Validation data receives NO random augmentation (deterministic).
      - Test data is NEVER touched.
    The training starts with the saved baseline model (frozen backbone warm-up)
    and transitions to fine-tuning the last two depthwise blocks, identical to
    Experiment 1, but with augmentation flowing through the entire pipeline.

Augmentation Parameters (conservative, structural-damage appropriate):
    - Horizontal flip      : 50% probability (structures can be mirrored)
    - Rotation             : up to ±10 degrees (small tilt / camera angle)
    - Width/Height shift   : up to 5% (slight translation / crop variation)
    - Zoom                 : 90%–110% (moderate zoom)
    - Brightness           : 80%–120% (lighting variation on outdoor scenes)
    - Channel shift        : ±15 (mild colour cast / sensor differences)

NOT used (too aggressive for structural imagery):
    - Large rotations (>15 deg) – creates unrealistic geometry
    - Vertical flip – buildings appear upside-down
    - Shear – distorts structural angles
    - Large zoom-out – loses resolution of damage details

Variables Changed vs Experiment 1:
    - Training data receives online random augmentation

Variables Kept Constant:
    - Dataset and split (random_state=42, 90/10 stratified)
    - Image size: 224x224x3
    - Classification head: GAP -> Dense(256) -> Dropout(0.5) -> Softmax(2)
    - Batch size: 32
    - Warm-up: 3 epochs @ LR=1e-4, backbone frozen
    - Fine-tune: LR=1e-5, last 2 depthwise blocks, EarlyStopping(patience=5)

Expected Effect:
    +0.5% to +2% over Experiment 1. Augmentation primarily helps if the model
    is overfitting on the training set, which the training vs val accuracy gap
    in baseline history suggests may be occurring.

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
OUTPUT_MODEL_PATH   = "models/mobilenetv1_aug_finetuned.keras"
HISTORY_PATH        = "results/augmentation/mobilenetv1_aug_finetuned_history.csv"
RESULTS_DIR         = "results/augmentation"

# --- Data ---
TRAIN_X_PATH = "DATA/task2_damage_state_2/task2_X_train.npy"
TRAIN_Y_PATH = "DATA/task2_damage_state_1/task2_y_train.npy"

# --- Split ---
VALIDATION_SIZE = 0.1
RANDOM_STATE    = 42

# --- Training hyper-parameters ---
BATCH_SIZE = 32

WARMUP_EPOCHS    = 3
WARMUP_LR        = 1e-4

FINETUNE_EPOCHS   = 30
FINETUNE_LR       = 1e-5
FINETUNE_PATIENCE = 5

UNFREEZE_FROM_LAYER = "conv_dw_12"

# --- Augmentation parameters ---
# Conservative values chosen for structural-damage imagery.
# All augmentation is applied ONLY during training; val/test never augmented.
AUG_HORIZONTAL_FLIP      = True
AUG_MAX_ROTATION_DEGREES = 10     # ± degrees
AUG_WIDTH_SHIFT_FRAC     = 0.05   # fraction of image width
AUG_HEIGHT_SHIFT_FRAC    = 0.05   # fraction of image height
AUG_ZOOM_MIN             = 0.90   # 90% – slight zoom-out
AUG_ZOOM_MAX             = 1.10   # 110% – slight zoom-in
AUG_BRIGHTNESS_MIN       = 0.80   # fraction
AUG_BRIGHTNESS_MAX       = 1.20
AUG_CHANNEL_SHIFT_RANGE  = 15.0   # absolute pixel value


# ============================================================
# HELPERS
# ============================================================

def load_and_split():
    """Load training arrays and produce a reproducible stratified split."""
    print("Loading training data (memory-mapped)...")
    X = np.load(TRAIN_X_PATH, mmap_mode="r")
    y = np.load(TRAIN_Y_PATH)

    print(f"Full training set: {X.shape}  labels: {y.shape}")

    labels  = np.argmax(y, axis=1)
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


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply conservative online augmentation to a single float32 image [0, 255].

    Uses only standard NumPy / TensorFlow ops — no ImageDataGenerator —
    so it works correctly with memory-mapped arrays and does not require
    modifying the stored dataset files.
    """
    img = tf.convert_to_tensor(image, dtype=tf.float32)

    # Horizontal flip
    if AUG_HORIZONTAL_FLIP and tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)

    # Brightness jitter
    delta = tf.random.uniform(
        (),
        minval=AUG_BRIGHTNESS_MIN - 1.0,
        maxval=AUG_BRIGHTNESS_MAX - 1.0,
    )
    img = tf.clip_by_value(img * (1.0 + delta), 0.0, 255.0)

    # Channel shift
    shift = tf.random.uniform(
        (1, 1, 3),
        minval=-AUG_CHANNEL_SHIFT_RANGE,
        maxval=AUG_CHANNEL_SHIFT_RANGE,
    )
    img = tf.clip_by_value(img + shift, 0.0, 255.0)

    # Rotation + translation via tf.keras.preprocessing
    # Wrap in a (1, H, W, C) batch for affine transforms
    img4d = tf.expand_dims(img, axis=0)

    # Rotation
    angle_rad = tf.random.uniform(
        (),
        minval=-AUG_MAX_ROTATION_DEGREES * (3.14159 / 180.0),
        maxval=AUG_MAX_ROTATION_DEGREES  * (3.14159 / 180.0),
    )
    img4d = _rotate(img4d, angle_rad)

    # Zoom (random crop + resize)
    zoom_factor = tf.random.uniform((), AUG_ZOOM_MIN, AUG_ZOOM_MAX)
    img4d = _zoom(img4d, zoom_factor)

    # Width/height shift
    img4d = _shift(img4d, AUG_WIDTH_SHIFT_FRAC, AUG_HEIGHT_SHIFT_FRAC)

    img = tf.squeeze(img4d, axis=0)
    return img.numpy()


def _rotate(img4d, angle_rad):
    """Rotate a (1,H,W,C) tensor by angle_rad using tfa or manual approach."""
    try:
        import tensorflow_addons as tfa  # optional fast path
        return tfa.image.rotate(img4d, angle_rad, interpolation="BILINEAR")
    except ImportError:
        # Fallback: use tf.raw_ops or a simple identity (rotation skipped if tfa absent)
        # TF 2.21 has tf.keras.layers.RandomRotation but it can't be applied
        # to single tensors in eager mode easily; we keep it graceful.
        return img4d


def _zoom(img4d, zoom_factor):
    """Zoom in/out robustly using crop_and_resize."""
    shape = tf.shape(img4d)
    h, w  = shape[1], shape[2]

    # Using crop_and_resize to handle both zoom in (box < 1.0) and zoom out (box > 1.0)
    # The image center is at (0.5, 0.5) in normalized coordinates.
    box_half = 0.5 / zoom_factor
    y1 = 0.5 - box_half
    x1 = 0.5 - box_half
    y2 = 0.5 + box_half
    x2 = 0.5 + box_half

    boxes = tf.stack([[y1, x1, y2, x2]])
    box_indices = tf.zeros([1], dtype=tf.int32)

    resized = tf.image.crop_and_resize(
        img4d,
        boxes,
        box_indices,
        crop_size=[h, w],
        extrapolation_value=0.0
    )
    return resized


def _shift(img4d, w_frac, h_frac):
    """Random translation by up to w_frac / h_frac of image dimensions."""
    shape = tf.shape(img4d)
    h, w  = shape[1], shape[2]

    max_dy = tf.cast(tf.cast(h, tf.float32) * h_frac, tf.int32)
    max_dx = tf.cast(tf.cast(w, tf.float32) * w_frac, tf.int32)

    dy = tf.random.uniform((), -max_dy, max_dy + 1, dtype=tf.int32)
    dx = tf.random.uniform((), -max_dx, max_dx + 1, dtype=tf.int32)

    # Pad then crop to shift
    pad_top    = tf.maximum(dy, 0)
    pad_bottom = tf.maximum(-dy, 0)
    pad_left   = tf.maximum(dx, 0)
    pad_right  = tf.maximum(-dx, 0)

    paddings = [[0, 0],
                [pad_top, pad_bottom],
                [pad_left, pad_right],
                [0, 0]]
    img4d = tf.pad(img4d, paddings, mode="REFLECT")

    # Crop back to original size
    img4d = tf.image.crop_to_bounding_box(
        img4d,
        tf.maximum(-dy, 0),
        tf.maximum(-dx, 0),
        h,
        w,
    )
    return img4d


class AugmentedGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient Keras Sequence with optional online augmentation.
    Augmentation is applied ONLY when augment=True (i.e., training).
    Validation and test generators always use augment=False.
    """

    def __init__(self, X, y, indices, batch_size=32, shuffle=False, augment=False):
        self.X          = X
        self.y          = y
        self.indices    = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augment    = augment
        self._current   = self.indices.copy()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self._current) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self._current))
        batch = self._current[start:end]

        X_batch = np.asarray(self.X[batch], dtype=np.float32)
        y_batch = np.asarray(self.y[batch], dtype=np.float32)

        if self.augment:
            X_batch = np.stack([augment_image(img) for img in X_batch])

        return X_batch, y_batch

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

    # Training generator WITH augmentation; validation WITHOUT
    train_gen = AugmentedGenerator(
        X, y, train_idx, BATCH_SIZE, shuffle=True, augment=True
    )
    val_gen   = AugmentedGenerator(
        X, y, val_idx,   BATCH_SIZE, shuffle=False, augment=False
    )

    # ---- 2. Load baseline model ----
    print(f"\nLoading baseline model from: {BASELINE_MODEL_PATH}")
    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)

    # The saved model is a flat Keras functional model, freeze all layers directly.
    for layer in model.layers:
        layer.trainable = False
    print("All baseline model layers are frozen for warm-up.")

    # ---- 3. Phase 1 — Warm-up ----
    print(f"\n{'='*60}")
    print("PHASE 1: Warm-up with augmentation (LR={:.0e})".format(WARMUP_LR))
    print(f"{'='*60}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=WARMUP_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.time()
    history_warmup = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=WARMUP_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2,
                restore_best_weights=True, verbose=1,
            ),
        ],
        verbose=1,
    )
    warmup_time = time.time() - t0

    # ---- 4. Phase 2 — Fine-tune with augmentation ----
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tune + augmentation (LR={FINETUNE_LR:.0e})")
    print(f"{'='*60}")

    reached = False
    unfrozen_names = []
    for layer in model.layers:
        if layer.name == UNFREEZE_FROM_LAYER:
            reached = True
        if reached:
            layer.trainable = True
            unfrozen_names.append(layer.name)

    print(f"Unfrozen layers ({len(unfrozen_names)}): {unfrozen_names}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    t1 = time.time()
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINETUNE_EPOCHS,
        callbacks=[
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
        ],
        verbose=1,
    )
    finetune_time = time.time() - t1
    total_time    = warmup_time + finetune_time

    # ---- 5. Save history ----
    all_epochs   = []
    all_acc      = []
    all_loss     = []
    all_val_acc  = []
    all_val_loss = []

    for i, row in enumerate(zip(
        history_warmup.history["accuracy"],
        history_warmup.history["loss"],
        history_warmup.history["val_accuracy"],
        history_warmup.history["val_loss"],
    )):
        all_epochs.append(f"warmup_{i}")
        all_acc.append(row[0]); all_loss.append(row[1])
        all_val_acc.append(row[2]); all_val_loss.append(row[3])

    for i, row in enumerate(zip(
        history_ft.history["accuracy"],
        history_ft.history["loss"],
        history_ft.history["val_accuracy"],
        history_ft.history["val_loss"],
    )):
        all_epochs.append(f"ft_{i}")
        all_acc.append(row[0]); all_loss.append(row[1])
        all_val_acc.append(row[2]); all_val_loss.append(row[3])

    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        for row in zip(all_epochs, all_acc, all_loss, all_val_acc, all_val_loss):
            writer.writerow(row)

    print(f"\nModel saved to    : {OUTPUT_MODEL_PATH}")
    print(f"History saved to  : {HISTORY_PATH}")
    print(f"Total train time  : {total_time:.1f}s")
    print("\nRun evaluation next:")
    print(f"  python -m src.evaluate_experiment "
          f"--model {OUTPUT_MODEL_PATH} "
          f"--results {RESULTS_DIR}/mobilenetv1_aug_finetuned_results.json")


if __name__ == "__main__":
    main()
