"""
================================================================================
Shared Experiment Evaluation Script
================================================================================

Purpose:
    Evaluate any trained .keras model on the UNTOUCHED test set and save
    results as JSON.  Used consistently across all three experiments so that
    metrics are computed identically for fair comparison against the 76.78%
    MobileNetV1 baseline.

Usage:
    python -m src.evaluate_experiment \
        --model  models/mobilenetv1_finetuned.keras \
        --results results/finetuning/mobilenetv1_finetuned_results.json

    python -m src.evaluate_experiment \
        --model  models/mobilenetv1_aug_finetuned.keras \
        --results results/augmentation/mobilenetv1_aug_finetuned_results.json

    python -m src.evaluate_experiment \
        --model  models/mobilenetv1_gradcam_region.keras \
        --results results/gradcam_region/mobilenetv1_gradcam_region_results.json

DO NOT EXECUTE during implementation; run manually after training each model.
================================================================================
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

X_TEST_PATH = "DATA/task2_damage_state_1/task2_X_test.npy"
Y_TEST_PATH = "DATA/task2_damage_state_1/task2_y_test.npy"

CLASS_NAMES = ["undamaged", "damaged"]

# Baseline reference for delta reporting
BASELINE_ACCURACY  = 0.7678
BASELINE_PRECISION = 0.7435
BASELINE_RECALL    = 0.8028
BASELINE_F1        = 0.7720


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Keras model on the structural-damage test set"
    )
    parser.add_argument("--model",   required=True,
                        help="Path to the .keras model file to evaluate")
    parser.add_argument("--results", required=True,
                        help="Path to save the JSON results file")
    parser.add_argument("--batch",   type=int, default=32,
                        help="Batch size for inference (default: 32)")
    args = parser.parse_args()

    results_dir = os.path.dirname(args.results)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    print("Model loaded.")

    # ---- Load test set ----
    print("\nLoading untouched test set...")
    X_test     = np.load(X_TEST_PATH, mmap_mode="r")
    y_test_oh  = np.load(Y_TEST_PATH)
    y_test     = np.argmax(y_test_oh, axis=1)

    print(f"Test images : {X_test.shape}")
    print(f"Test labels : {y_test.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ---- Inference ----
    print("\nRunning inference on test set...")
    t0     = time.time()
    y_prob = model.predict(
        X_test.astype(np.float32),
        batch_size=args.batch,
        verbose=1,
    )
    inference_time = time.time() - t0
    print(f"Inference time: {inference_time:.2f}s")

    y_pred = np.argmax(y_prob, axis=1)

    # ---- Metrics ----
    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    cm        = confusion_matrix(y_test, y_pred).tolist()
    report    = classification_report(y_test, y_pred,
                                       target_names=CLASS_NAMES, zero_division=0)

    # ---- Deltas vs baseline ----
    d_acc  = accuracy  - BASELINE_ACCURACY
    d_prec = precision - BASELINE_PRECISION
    d_rec  = recall    - BASELINE_RECALL
    d_f1   = f1        - BASELINE_F1

    # ---- Print results ----
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Model: {args.model}")
    print(sep)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)  "
          f"Δ = {d_acc:+.4f} ({d_acc*100:+.2f}%)")
    print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)  "
          f"Δ = {d_prec:+.4f} ({d_prec*100:+.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)  "
          f"Δ = {d_rec:+.4f} ({d_rec*100:+.2f}%)")
    print(f"  F1 Score  : {f1:.4f}  ({f1*100:.2f}%)  "
          f"Δ = {d_f1:+.4f} ({d_f1*100:+.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    print(f"\n  Classification Report:")
    print(report)
    print(sep)

    # ---- Save results ----
    results = {
        "model_path":          args.model,
        "test_images":         int(len(X_test)),
        "baseline": {
            "accuracy":  BASELINE_ACCURACY,
            "precision": BASELINE_PRECISION,
            "recall":    BASELINE_RECALL,
            "f1_score":  BASELINE_F1,
        },
        "experiment": {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1,
        },
        "delta_vs_baseline": {
            "accuracy":  round(d_acc,  4),
            "precision": round(d_prec, 4),
            "recall":    round(d_rec,  4),
            "f1_score":  round(d_f1,   4),
        },
        "confusion_matrix":    cm,
        "inference_time_s":    round(inference_time, 2),
    }

    with open(args.results, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {args.results}")


if __name__ == "__main__":
    main()
