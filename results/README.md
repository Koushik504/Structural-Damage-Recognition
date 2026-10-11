# Results

This folder contains the results generated during the Structural Damage Recognition project.

## Model Comparison

The final experiments compare classical machine learning classifiers using 2048-dimensional InceptionV3 features with an end-to-end MobileNetV1 model.

### Test Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| KNN | 61.44% | 62.75% | 52.31% | 57.06% |
| Logistic Regression | 64.32% | 64.92% | 59.02% | 61.83% |
| RBF-SVM | 65.96% | 68.41% | 56.64% | 61.97% |
| InceptionV3 | 65.75% | 69.30% | 53.99% | 60.69% |
| **MobileNetV1** | **76.78%** | **74.35%** | **80.28%** | **77.20%** |

### Best Performing Model

**MobileNetV1** achieved the best overall performance with:

- Accuracy: **76.78%**
- Precision: **74.35%**
- Recall: **80.28%**
- F1-score: **77.20%**

MobileNetV1 outperformed the other evaluated models in both test accuracy and F1-score.

### Model Comparison Visualizations

- `results/model_comparison.csv`
- `results/model_accuracy_comparison.png`
- `results/model_f1_comparison.png`

---

## InceptionV3 Final Evaluation

The final InceptionV3 model was trained using:

- 90% of the training data for training
- 10% of the training data for validation
- The test set was kept separate and used only for final evaluation
- Batch size: 16
- Maximum epochs: 10
- Early stopping based on validation accuracy
- Best model checkpoint selected using validation accuracy

### InceptionV3 Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 65.75% |
| Precision | 69.30% |
| Recall | 53.99% |
| F1-score | 60.69% |
| Test samples | 1460 |

### InceptionV3 Confusion Matrix

Predicted:

| Actual | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 574 | 171 |
| Class 1 | 329 | 386 |

The confusion matrix shows that the model correctly classified 574 samples from class 0 and 386 samples from class 1.
### InceptionV3 Training Results

Training history and plots are available in:

- results/training/history_final.csv
- results/training/accuracy_curve.png
- results/training/loss_curve.png

---

## Classical Machine Learning Results

The classical classifiers were trained using 2048-dimensional feature vectors extracted from the trained InceptionV3 model.

### KNN

- Accuracy: **61.44%**
- Precision: **62.75%**
- Recall: **52.31%**
- F1-score: **57.06%**

### Logistic Regression

- Accuracy: **64.32%**
- Precision: **64.92%**
- Recall: **59.02%**
- F1-score: **61.83%**

### RBF-SVM

- Accuracy: **65.96%**
- Precision: **68.41%**
- Recall: **56.64%**
- F1-score: **61.97%**

The RBF-SVM achieved the strongest performance among the classical classifiers.

---

## MobileNetV1 Results

MobileNetV1 was evaluated as an end-to-end deep learning model.

### Test Performance

| Metric | Result |
|---|---:|
| Accuracy | **76.78%** |
| Precision | **74.35%** |
| Recall | **80.28%** |
| F1-score | **77.20%** |

### Confusion Matrix

MobileNetV1 confusion matrix:

Actual 0: 547 correct, 198 incorrect
Actual 1: 574 correct, 141 incorrect

MobileNetV1 achieved the best overall performance among all evaluated models.

---

## Conclusion

The experiments show that the end-to-end MobileNetV1 model performed better than both the InceptionV3 classifier and the classical machine learning approaches based on InceptionV3 features.

MobileNetV1 achieved **76.78% accuracy** and a **77.20% F1-score**, making it the best-performing model in this experiment.

The results suggest that the lightweight MobileNetV1 architecture was able to learn useful structural damage representations directly from the input images while maintaining a better balance between precision and recall than the other evaluated approaches.

---

## Accuracy Improvement Experiments

Three controlled experiments designed to push beyond the **76.78% baseline**.

### Experiment 1 — MobileNetV1 Fine-Tuning

| Property             | Value                                        |
|----------------------|----------------------------------------------|
| Script               | `src/train_mobilenetv1_finetuned.py`         |
| Output model         | `models/mobilenetv1_finetuned.keras`         |
| Phase 1 (warm-up)    | 3 epochs, LR=1e-4, backbone frozen           |
| Phase 2 (fine-tune)  | max 30 epochs, LR=1e-5, conv_dw_12+ unfrozen |
| EarlyStopping        | patience=5, monitor=val_loss                 |
| Hypothesis           | Partial backbone fine-tuning improves feature adaptation |

**Run command:**
```powershell
python -m src.train_mobilenetv1_finetuned
python -m src.evaluate_experiment --model models/mobilenetv1_finetuned.keras --results results/finetuning/mobilenetv1_finetuned_results.json
```

| Metric    | Baseline | Exp 1  | Δ   |
|-----------|----------|--------|-----|
| Accuracy  | 76.78%   | TBD    | TBD |
| Precision | 74.35%   | TBD    | TBD |
| Recall    | 80.28%   | TBD    | TBD |
| F1        | 77.20%   | TBD    | TBD |

---

### Experiment 2 — Data Augmentation + Fine-Tuning

| Property             | Value                                            |
|----------------------|--------------------------------------------------|
| Script               | `src/train_mobilenetv1_aug_finetuned.py`         |
| Output model         | `models/mobilenetv1_aug_finetuned.keras`         |
| Augmentation         | Flip, ±10° rotation, ±5% shift, 90-110% zoom, brightness 80-120%, channel shift ±15 |
| Phase 1 (warm-up)    | 3 epochs, LR=1e-4, backbone frozen              |
| Phase 2 (fine-tune)  | max 30 epochs, LR=1e-5, conv_dw_12+ unfrozen   |
| Hypothesis           | Augmentation reduces overfitting, augments Exp 1 gains |

**Run command:**
```powershell
python -m src.train_mobilenetv1_aug_finetuned
python -m src.evaluate_experiment --model models/mobilenetv1_aug_finetuned.keras --results results/augmentation/mobilenetv1_aug_finetuned_results.json
```

| Metric    | Baseline | Exp 2  | Δ   |
|-----------|----------|--------|-----|
| Accuracy  | 76.78%   | TBD    | TBD |
| Precision | 74.35%   | TBD    | TBD |
| Recall    | 80.28%   | TBD    | TBD |
| F1        | 77.20%   | TBD    | TBD |

---

### Experiment 3 — Grad-CAM Region Training

| Property             | Value                                              |
|----------------------|----------------------------------------------------|
| Grad-CAM script      | `src/gradcam.py`                                   |
| Preprocessing script | `src/region_focused_preprocessing.py`              |
| Training script      | `src/train_gradcam_region_model.py`                |
| Output model         | `models/mobilenetv1_gradcam_region.keras`          |
| Crops per image      | 5 (centre + 4 corners, 160×160 → 224×224)         |
| Training samples     | 10630 × 5 = 53 150                                 |
| Target layer         | `conv_pw_13_relu`                                  |
| Hypothesis           | Multi-crop training from spatially-varied regions improves accuracy |

**Run commands (in order):**
```powershell
# Step 1 – Grad-CAM audit
python -m src.gradcam --model models/mobilenetv1.keras --n 50 --outdir results/gradcam_region/heatmaps --seed 42

# Step 2 – Build region dataset (~15-40 min)
python -m src.region_focused_preprocessing --outdir results/gradcam_region/preprocessed

# Step 3 – Train
python -m src.train_gradcam_region_model

# Step 4 – Evaluate
python -m src.evaluate_experiment --model models/mobilenetv1_gradcam_region.keras --results results/gradcam_region/mobilenetv1_gradcam_region_results.json
```

| Metric    | Baseline | Exp 3  | Δ   |
|-----------|----------|--------|-----|
| Accuracy  | 76.78%   | TBD    | TBD |
| Precision | 74.35%   | TBD    | TBD |
| Recall    | 80.28%   | TBD    | TBD |
| F1        | 77.20%   | TBD    | TBD |

