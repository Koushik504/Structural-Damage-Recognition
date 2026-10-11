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

Controlled experiments designed to explore improvements over the **76.78% baseline** MobileNetV1 model.

The baseline model (`models/mobilenetv1.keras`) remains unchanged.

### Experiment 1 — MobileNetV1 Fine-Tuning

Partial fine-tuning of top layers in the MobileNetV1 backbone to adapt high-level convolutional features.

- **Hypothesis**: Partial backbone fine-tuning improves feature adaptation to structural damage patterns.
- **Model Path**: `models/mobilenetv1_finetuned.keras`
- **Training Script**: `src/train_mobilenetv1_finetuned.py`
- **Training History**: `results/training/mobilenetv1_finetuned_history.csv`
- **Evaluation JSON**: `results/finetuning/mobilenetv1_finetuned_test_results.json`
- **Configuration**:
  - Phase 1 (warm-up): 3 epochs, LR = 1e-4, backbone frozen
  - Phase 2 (fine-tune): up to 30 epochs, LR = 1e-5, `conv_dw_12` and subsequent layers unfrozen
  - EarlyStopping: patience = 5, monitor = `val_loss`

#### Verified Test Results (1,460 samples)
- **Accuracy**: 77.6712% (77.67%)
- **Precision**: 77.7461% (77.75%)
- **Recall**: 76.2238% (76.22%)
- **F1-score**: 76.9774% (76.98%)
- **Confusion Matrix**: `[[589, 156], [170, 545]]`

#### Delta vs. Baseline
- Accuracy: **+0.89 percentage points**
- Precision: **+3.40 percentage points**
- Recall: **-4.06 percentage points**
- F1-score: **-0.22 percentage points**

*Interpretation*: Experiment 1 achieves higher overall accuracy (+0.89%) and precision (+3.40%) compared to the baseline, but exhibits a modest trade-off with lower recall (-4.06%) and slightly lower F1-score (-0.22%).

---

### Experiment 3 — Grad-CAM Region-Focused MobileNetV1

Spatially focused training utilizing multi-crop extraction informed by visual activation regions.

- **Hypothesis**: Multi-crop training from spatially-varied regions improves localization and recognition accuracy.
- **Model Path**: `models/mobilenetv1_gradcam_region.keras`
- **Grad-CAM Audit Script**: `src/gradcam.py`
- **Preprocessing Script**: `src/region_focused_preprocessing.py`
- **Training Script**: `src/train_gradcam_region_model.py`
- **Evaluation Script**: `src/evaluate_experiment.py`
- **Training History**: `results/gradcam_region/mobilenetv1_gradcam_region_history.csv`
- **Evaluation JSON**: `results/gradcam_region/mobilenetv1_gradcam_region_results.json`
- **Configuration**:
  - Crops per image: 5 (centre + 4 corners, 160×160 resized to 224×224)
  - Target layer for Grad-CAM audit: `conv_pw_13_relu`
  - Training completed across 15 epochs

#### Verified Test Results (1,460 samples)
- **Accuracy**: 75.6164% (75.62%)
- **Precision**: 73.8380% (73.84%)
- **Recall**: 77.7622% (77.76%)
- **F1-score**: 75.7493% (75.75%)
- **Confusion Matrix**: `[[548, 197], [159, 556]]`
- **Inference Time**: 113.13 seconds

#### Delta vs. Baseline
- Accuracy: **-1.16 percentage points**
- Precision: **-0.51 percentage points**
- Recall: **-2.52 percentage points**
- F1-score: **-1.45 percentage points**

*Interpretation*: Experiment 3 does not outperform the baseline on any of the four evaluated classification metrics. Training on cropped regions introduced border context loss and spatial artifacting that degraded general classification performance.

---

### Results Comparison Table

| Metric | Baseline | Experiment 1 | Experiment 3 |
|---|---:|---:|---:|
| Accuracy | 76.78% | 77.67% | 75.62% |
| Precision | 74.35% | 77.75% | 73.84% |
| Recall | 80.28% | 76.22% | 77.76% |
| F1-score | 77.20% | 76.98% | 75.75% |
| Test samples | 1,460 | 1,460 | 1,460 |

### Summary & Limitations
- **Baseline Integrity**: The baseline MobileNetV1 model (`models/mobilenetv1.keras`) remains completely unchanged.
- **Experiment 1**: Achieved the highest accuracy (77.67%) and precision (77.75%) among evaluated MobileNetV1 variants, but sacrifices recall (-4.06%).
- **Experiment 3**: Did not improve overall metrics compared to the baseline, demonstrating that uniform multi-crop regional focus alone is insufficient for improving damage classification.

---

## Final Comparison of All Six Accuracy Improvement Experiments

All six experiments were evaluated using the same held-out test set of **1,460 structural images**. The original MobileNetV1 model serves as the baseline.

### Complete Test Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| MobileNetV1 Baseline | 76.78% | 74.35% | 80.28% | 77.20% |
| Experiment 1 — MobileNetV1 Fine-Tuning | 77.67% | 77.75% | 76.22% | 76.98% |
| Experiment 2 — Data Augmentation + Fine-Tuning | 77.33% | 77.35% | 75.94% | 76.64% |
| Experiment 3 — Grad-CAM Region Training | 75.62% | 73.84% | 77.76% | 75.75% |
| **Experiment 4 — EfficientNetB0** | **87.26%** | 86.99% | **86.99%** | **86.99%** |
| Experiment 5 — EfficientNetB0 Fine-Tuning | **87.26%** | **87.95%** | 85.73% | 86.83% |
| Experiment 6 — Weighted Ensemble | 77.40% | 75.50% | 79.72% | 77.55% |

**Note:** Precision, recall, and F1-score use class 1 (Undamaged) as the positive class. Class 0 represents Damaged images.

### Accuracy Improvement Over Baseline

| Experiment | Accuracy Change |
|---|---:|
| Experiment 1 | +0.89 percentage points |
| Experiment 2 | +0.55 percentage points |
| Experiment 3 | -1.16 percentage points |
| Experiment 4 | **+10.48 percentage points** |
| Experiment 5 | **+10.48 percentage points** |
| Experiment 6 | +0.62 percentage points |

### Key Findings

- **Experiment 4 (EfficientNetB0)** achieved the best overall performance, with 87.26% test accuracy and 86.99% F1-score.
- **Experiment 5 (EfficientNetB0 Fine-Tuning)** matched Experiment 4's accuracy and achieved the highest precision (87.95%), but its F1-score was slightly lower.
- **Experiment 1** improved MobileNetV1 accuracy through partial fine-tuning, although recall and F1-score decreased.
- **Experiment 2** produced a modest accuracy improvement using data augmentation and fine-tuning.
- **Experiment 3** performed below the baseline, indicating that the region-focused training approach did not improve classification performance in this experiment.
- **Experiment 6** improved slightly over the MobileNetV1 baseline using a weighted combination of MobileNetV1 and InceptionV3 predictions.

### Final Conclusion

The experiments demonstrate that the choice of model architecture had a greater impact on classification accuracy than the tested fine-tuning, augmentation, region-focused training, and ensemble strategies.

**EfficientNetB0 (Experiment 4) is the recommended final model**, achieving **87.26% accuracy**, an improvement of **10.48 percentage points** over the original MobileNetV1 baseline.

The baseline models and their historical evaluation results remain preserved for reproducibility and comparison.
