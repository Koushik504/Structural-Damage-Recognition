
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
