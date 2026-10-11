"""
================================================================================
Grad-CAM Utility — Structural Damage MobileNetV1
================================================================================

Purpose:
    Generate Grad-CAM heatmaps and overlays for a sample of test images using
    the baseline MobileNetV1 model.  The outputs are used to visually audit
    WHERE the model attends when making predictions, informing the region-
    focused preprocessing strategy in Experiment 3.

Target layer:
    conv_pw_13_relu  —  the final convolutional activation in MobileNetV1
    (pointwise convolution of block 13).  This layer has spatial extent and
    is the last feature map before GlobalAveragePooling, making it ideal for
    Grad-CAM.

Usage (DO NOT run during implementation; run manually):
    python -m src.gradcam \
        --model  models/mobilenetv1.keras \
        --n      50 \
        --outdir results/gradcam_region/heatmaps \
        --seed   42

Outputs (all in results/gradcam_region/heatmaps/):
    <idx>_<true>_<pred>_original.png   — original RGB image
    <idx>_<true>_<pred>_heatmap.png    — raw Grad-CAM heatmap (jet colourmap)
    <idx>_<true>_<pred>_overlay.png    — heatmap blended with original image
    gradcam_metadata.csv               — per-image prediction metadata

DO NOT EXECUTE this script; I will run it manually.
================================================================================
"""

import argparse
import csv
import os

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION DEFAULTS
# ============================================================

DEFAULT_MODEL_PATH = "models/mobilenetv1.keras"
DEFAULT_N_SAMPLES  = 50
DEFAULT_OUTDIR     = "results/gradcam_region/heatmaps"
DEFAULT_SEED       = 42

# The final conv activation layer in MobileNetV1
GRADCAM_LAYER_NAME = "conv_pw_13_relu"

# Test data paths
X_TEST_PATH = "DATA/task2_damage_state_1/task2_X_test.npy"
Y_TEST_PATH = "DATA/task2_damage_state_1/task2_y_test.npy"

CLASS_NAMES = ["damaged", "undamaged"]


# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================

def make_gradcam_model(model: tf.keras.Model, layer_name: str):
    """
    Build a Grad-CAM sub-model that outputs:
        (final_conv_activations, model_predictions)
    from the same input as the original model.
    """
    # Locate the target layer by name
    conv_layer = model.get_layer(layer_name)

    # If model is nested (e.g., backbone is a sub-model), search recursively
    if conv_layer is None:
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    if sub.name == layer_name:
                        conv_layer = sub
                        break
            if conv_layer is not None:
                break

    if conv_layer is None:
        raise ValueError(
            f"Layer '{layer_name}' not found in model. "
            f"Available layers: {[l.name for l in model.layers]}"
        )

    gradcam_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )
    return gradcam_model


def compute_gradcam(gradcam_model, image: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single image.

    Args:
        gradcam_model : model with outputs [conv_activations, predictions]
        image         : float32 array of shape (H, W, C), values in [0, 255]
        class_idx     : the class index to compute gradients for

    Returns:
        heatmap : float32 array of shape (H, W), values normalised to [0, 1]
    """
    img_tensor = tf.expand_dims(
        tf.cast(image, tf.float32), axis=0
    )  # (1, H, W, C)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_tensor)
        # Watch the conv output tensor
        tape.watch(conv_outputs)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)  # (1, h, w, filters)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)

    # Weight each channel of the activation map
    conv_outputs = conv_outputs[0]  # (h, w, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)  # (h, w)

    # ReLU and normalise
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()

    h_max = heatmap.max()
    if h_max > 0:
        heatmap /= h_max

    return heatmap.astype(np.float32)


def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap with the original image.

    Args:
        original_rgb : uint8 (H, W, 3) in RGB
        heatmap      : float32 (H, W) in [0, 1]
        alpha        : heatmap opacity

    Returns:
        overlay : uint8 (H, W, 3) blended image
    """
    h, w = original_rgb.shape[:2]

    # Resize heatmap to image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply jet colourmap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

    # Blend
    original_float  = original_rgb.astype(np.float32)
    heatmap_float   = heatmap_rgb.astype(np.float32)
    overlay = (1 - alpha) * original_float + alpha * heatmap_float
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_image(path: str, img_rgb: np.ndarray):
    """Save an RGB image using OpenCV (converts to BGR)."""
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for MobileNetV1"
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH,
                        help="Path to the .keras model file")
    parser.add_argument("--layer",  default=GRADCAM_LAYER_NAME,
                        help="Target convolutional layer for Grad-CAM")
    parser.add_argument("--n",      type=int, default=DEFAULT_N_SAMPLES,
                        help="Number of test images to visualise")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Directory to write heatmap images and metadata")
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    print("Model loaded.")

    # Try to find the target layer — handle nested sub-model
    layer_names_top = [l.name for l in model.layers]
    print(f"Top-level layers: {layer_names_top}")

    try:
        gradcam_model = make_gradcam_model(model, args.layer)
        print(f"Grad-CAM model built with target layer: {args.layer}")
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nSearching for similar layer names...")
        all_names = []
        for layer in model.layers:
            all_names.append(layer.name)
            if hasattr(layer, "layers"):
                all_names.extend(sub.name for sub in layer.layers)
        conv_like = [n for n in all_names if "conv" in n.lower() or "relu" in n.lower()]
        print("Conv/ReLU layers found:", conv_like[-10:])
        raise

    # ---- Load test data ----
    print(f"\nLoading test data...")
    X_test = np.load(X_TEST_PATH, mmap_mode="r")
    y_test_oh = np.load(Y_TEST_PATH)
    y_test = np.argmax(y_test_oh, axis=1)

    print(f"Test images: {X_test.shape}")

    # ---- Sample selection ----
    rng = np.random.default_rng(args.seed)
    n   = min(args.n, len(X_test))
    sample_indices = rng.choice(len(X_test), size=n, replace=False)

    # ---- CSV metadata writer ----
    meta_path = os.path.join(args.outdir, "gradcam_metadata.csv")
    meta_rows = []

    print(f"\nGenerating Grad-CAM for {n} images...")
    for rank, idx in enumerate(sample_indices):
        image   = np.asarray(X_test[idx], dtype=np.float32)   # (224, 224, 3)
        true_cl = int(y_test[idx])

        # Run prediction
        img_t   = tf.expand_dims(tf.cast(image, tf.float32), 0)
        probs   = model.predict(img_t, verbose=0)[0]
        pred_cl = int(np.argmax(probs))
        confidence = float(probs[pred_cl])

        # Grad-CAM for PREDICTED class (model's own attention)
        heatmap = compute_gradcam(gradcam_model, image, pred_cl)

        # Prepare output filenames
        prefix = f"{rank:04d}_true{true_cl}_pred{pred_cl}"
        orig_path    = os.path.join(args.outdir, f"{prefix}_original.png")
        heatmap_path = os.path.join(args.outdir, f"{prefix}_heatmap.png")
        overlay_path = os.path.join(args.outdir, f"{prefix}_overlay.png")

        # Original image (clip to uint8)
        original_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        # Heatmap as colourmap
        h_resized = cv2.resize(heatmap, (224, 224))
        h_uint8   = np.uint8(255 * h_resized)
        h_colour  = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        h_rgb     = cv2.cvtColor(h_colour, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = overlay_heatmap(original_uint8, heatmap)

        # Save
        save_image(orig_path,    original_uint8)
        save_image(heatmap_path, h_rgb)
        save_image(overlay_path, overlay)

        # Record metadata
        meta_rows.append({
            "rank":           rank,
            "test_index":     idx,
            "true_class":     CLASS_NAMES[true_cl],
            "pred_class":     CLASS_NAMES[pred_cl],
            "confidence":     f"{confidence:.4f}",
            "correct":        str(true_cl == pred_cl),
            "heatmap_max":    f"{heatmap.max():.4f}",
            "heatmap_mean":   f"{heatmap.mean():.4f}",
        })

        if (rank + 1) % 10 == 0 or rank == n - 1:
            print(f"  [{rank+1}/{n}] idx={idx} true={CLASS_NAMES[true_cl]} "
                  f"pred={CLASS_NAMES[pred_cl]} conf={confidence:.3f}")

    # Write CSV
    with open(meta_path, "w", newline="") as f:
        fieldnames = list(meta_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\nDone. {n} heatmaps saved to: {args.outdir}")
    print(f"Metadata CSV saved to      : {meta_path}")
    print("\nNext step: inspect heatmaps, then run region_focused_preprocessing.py")


if __name__ == "__main__":
    main()
