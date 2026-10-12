
# Structural Damage Recognition Using Machine Learning and Deep Learning

## 1. Project Overview

This project focuses on the automated recognition of structural damage from images using machine learning and deep learning techniques.

The objective is to classify structural images into two categories:

- **Damaged (Class 0)**
- **Undamaged (Class 1)**

The project includes six original baseline configurations and six additional accuracy-improvement experiments, making a total of 12 evaluated configurations.

The experiments investigate transfer learning, CNN-based feature extraction, fine-tuning, data augmentation, Grad-CAM-based region-focused learning, EfficientNetB0, and weighted ensemble classification.

The best overall performance was achieved by **EfficientNetB0 (Experiment 4)**, with **87.26% test accuracy** and **86.99% F1-score**.

## 2. Dataset Description

The project uses a labeled image dataset for binary structural damage classification.

### 2.1 Classes

The dataset contains two classes:

- **Damaged (Class 0):** Images showing structural damage.
- **Undamaged (Class 1):** Images showing structures without visible damage.

### 2.2 Dataset Splitting

The training data was divided into:

- **90% training data**
- **10% validation data**

A fixed random seed of **42** was used to ensure reproducibility.

The original test dataset was kept separate from training and validation to provide an unbiased final evaluation.

### 2.3 Test Dataset Distribution

| Class | Number of Images |
|---|---:|
| Damaged (Class 0) | 745 |
| Undamaged (Class 1) | 715 |
| **Total** | **1,460** |

The same test dataset was used to compare the evaluated configurations consistently.

## 3. Methodology

The project follows a structured workflow for training and evaluating structural damage classification models.

### 3.1 Data Preparation

Images are organized into two classes: Damaged and Undamaged. The training dataset is split into training and validation subsets using a fixed random seed of 42. The independent test dataset is reserved for final evaluation.

### 3.2 Model Development

The project evaluates six original baseline configurations using machine learning and deep learning approaches.

Six additional experiments investigate methods for improving classification accuracy, including:

- MobileNetV1 fine-tuning
- Data augmentation with fine-tuning
- Grad-CAM-based region-focused learning
- EfficientNetB0 with frozen feature extraction layers
- EfficientNetB0 fine-tuning
- MobileNetV1 and InceptionV3 weighted ensemble

### 3.3 Model Evaluation

All configurations are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified test images.
- **Precision:** Proportion of images predicted as Undamaged that are actually Undamaged.
- **Recall:** Proportion of actual Undamaged images correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

Precision, recall, and F1-score use **Class 1 (Undamaged)** as the positive class.

## 4. Experimental Results

Six additional experiments were conducted to improve classification performance beyond the original baseline models.

### 4.1 Accuracy-Improvement Experiments

| Experiment | Method | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| 1 | MobileNetV1 Fine-Tuning | 77.67% | 77.75% | 76.22% | 76.98% |
| 2 | Augmentation + Fine-Tuning | 77.33% | 77.35% | 75.94% | 76.64% |
| 3 | Grad-CAM Region-Focused | 75.62% | 73.84% | 77.76% | 75.75% |
| 4 | EfficientNetB0 (Frozen Backbone) | **87.26%** | 86.99% | **86.99%** | **86.99%** |
| 5 | EfficientNetB0 Fine-Tuning | **87.26%** | **87.95%** | 85.73% | 86.83% |
| 6 | MobileNetV1 + InceptionV3 Weighted Ensemble | 77.40% | 75.50% | 79.72% | 77.55% |

### 4.2 Best Performing Model

**EfficientNetB0 (Experiment 4)** achieved the best overall performance:

- **Accuracy:** 87.26%
- **Precision:** 86.99%
- **Recall:** 86.99%
- **F1-score:** 86.99%

Experiment 5 matched Experiment 4 in accuracy but achieved a slightly lower F1-score.

The best original baseline model, MobileNetV1, achieved 76.78% accuracy and 77.20% F1-score. Experiment 4 improved test accuracy by **10.48 percentage points** compared with this baseline.

All precision, recall, and F1-score values are calculated for **Class 1 (Undamaged)**.

Detailed results are available in [results/README.md](../results/README.md).

## 5. Discussion and Key Findings

The experimental results highlight several important observations.

### 5.1 Performance of the Original Baseline

MobileNetV1 was the best-performing model among the six original baseline configurations, achieving 76.78% test accuracy and 77.20% F1-score.

### 5.2 Impact of Fine-Tuning and Augmentation

MobileNetV1 fine-tuning (Experiment 1) improved accuracy to 77.67%.

Combining data augmentation with fine-tuning (Experiment 2) achieved 77.33% accuracy, indicating that augmentation did not provide an additional improvement over Experiment 1 in this evaluation.

### 5.3 Grad-CAM Region-Focused Learning

Experiment 3 achieved 75.62% accuracy. Although it investigated region-focused learning, it did not outperform the original MobileNetV1 baseline.

### 5.4 EfficientNetB0 Performance

EfficientNetB0 produced the strongest results among the evaluated experiments.

Experiment 4 achieved 87.26% accuracy and 86.99% F1-score. Experiment 5 achieved the same accuracy but a slightly lower F1-score of 86.83%.

This suggests that fine-tuning did not improve overall test accuracy over the frozen-backbone EfficientNetB0 configuration in this evaluation.

### 5.5 Weighted Ensemble Performance

The MobileNetV1 and InceptionV3 weighted ensemble (Experiment 6) achieved 77.40% accuracy and 77.55% F1-score.

The ensemble slightly outperformed the original MobileNetV1 baseline in accuracy and F1-score but remained below EfficientNetB0.

### 5.6 Overall Finding

EfficientNetB0 (Experiment 4) is the recommended final model based on its combination of high accuracy and the best F1-score.

These conclusions are based on the independent test dataset of 1,460 images.

## 6. Conclusion and Future Work

### 6.1 Conclusion

This project successfully evaluated six original baseline configurations and six additional accuracy-improvement experiments for binary structural damage recognition.

Among the original baseline configurations, MobileNetV1 achieved the best performance with 76.78% test accuracy.

After conducting the additional experiments, **EfficientNetB0 (Experiment 4)** emerged as the best overall model, achieving:

- **Accuracy:** 87.26%
- **Precision:** 86.99%
- **Recall:** 86.99%
- **F1-score:** 86.99%

Compared with the best original baseline, EfficientNetB0 improved test accuracy by 10.48 percentage points.

The results demonstrate the effectiveness of transfer learning for image-based structural damage classification.

### 6.2 Future Work

Possible future improvements include:

- Further analysis of Grad-CAM visualizations to understand model predictions.
- Investigation of incorrectly classified damaged and undamaged images.
- Evaluation of additional CNN architectures.
- Exploration of alternative ensemble strategies.
- Testing model generalization on images from different environments and structural conditions.
- Development of a practical application for structural damage recognition.

### 6.3 Final Outcome

The project has completed all 12 planned configurations and experiments.

**EfficientNetB0 (Experiment 4) is selected as the final best-performing model based on test accuracy and F1-score.**

The complete experimental results and implementation details are available in the project repository.
