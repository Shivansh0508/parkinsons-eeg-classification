# Parkinson's Disease Detection using EEG and MRI

A comprehensive machine learning and deep learning framework for Parkinson's Disease (PD) detection using multiple neuroimaging and neurophysiological datasets.

---

## Overview

This repository contains three independent Parkinson's Disease classification pipelines developed using publicly available datasets:

1. **OpenNeuro EEG Dataset (DS007526)**
   - Resting-state EEG recordings
   - Healthy Controls (HC) vs Parkinson's Disease (PD)
   - Signal preprocessing, feature extraction, and machine learning classification

2. **PPMI MRI Dataset**
   - Structural T1-weighted MRI scans
   - PD vs Healthy Controls
   - Deep learning and radiomics-based classification

3. **BrainLAT MRI Dataset**
   - Brain MRI-based Parkinson's Disease detection
   - Feature extraction and classification pipeline
   - Benchmark comparison against published studies

---

## Repository Structure

```text
├── OpenNeuro_EEG/
│   
│
├── PPMI_MRI/
│ 
│
├── BrainLAT_MRI/
│ 

└── README.md
```

---

## Datasets

### OpenNeuro EEG Dataset (DS007526)

- Modality: EEG
- Task: Resting-State EEG
- Subjects: Parkinson's Disease and Healthy Controls
- Source: OpenNeuro

### PPMI Dataset

- Modality: MRI
- Source: Parkinson's Progression Markers Initiative (PPMI)
- Subjects: PD and Healthy Controls
- Used for MRI-based classification and benchmarking

### BrainLAT Dataset

- Modality: MRI
- Used for external validation and comparative analysis
- Supports development of robust PD detection pipelines

---

---

## Methodology

### EEG Pipeline

#### Preprocessing

- Bandpass Filtering
- Common Average Referencing (CAR)
- Artifact Rejection
- Epoch Segmentation
- Channel Standardization

#### Feature Extraction

- Band Power Features
  - Delta
  - Theta
  - Alpha
  - Beta
  - Gamma

- Relative Band Powers
- Spectral Ratios
- Spectral Entropy
- Permutation Entropy
- Hjorth Parameters
- Spectral Edge Frequency
- FOOOF Features
- Functional Connectivity (PLV)

#### Classification

- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Ensemble Learning

---

### MRI Pipeline

#### Preprocessing

- Skull Stripping
- Intensity Normalization
- Registration
- Resampling

#### Feature Engineering

- Radiomic Features
- Texture Features
- Shape Features
- Deep Learning Features

#### Classification

- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Ensemble Learning

---

### MRI Pipeline

#### Preprocessing

- Skull Stripping
- Intensity Normalization
- Registration
- Resampling

#### Feature Engineering

- Radiomic Features
- Texture Features
- Shape Features
- Deep Learning Features

#### Classification

- Random Forest
- XGBoost
- CNN
- ResNet
- Vision Transformers

---

## Results

| Dataset | Modality | Task |
|----------|----------|----------|
| OpenNeuro DS007526 | EEG | PD vs HC |
| PPMI | MRI | PD vs HC |
| BrainLAT | MRI | PD vs HC |

Performance metrics reported:

- Accuracy
- AUC-ROC
- Precision
- Recall
- F1 Score
- Sensitivity
- Specificity

---

## Research Goals

- Develop robust Parkinson's Disease detection systems.
- Compare EEG and MRI-based approaches.
- Benchmark against published literature.
- Investigate explainable AI for neurological disorder classification.
- Build reproducible and clinically relevant ML pipelines.

---

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- MNE
- XGBoost
- LightGBM
- PyTorch
- TensorFlow
- MONAI
- Nilearn
- Matplotlib

---

## Citation

If you use this repository in your research, please cite the corresponding datasets:

- OpenNeuro DS007526
- Parkinson's Progression Markers Initiative (PPMI)
- BrainLAT

---

## Author

**Shivansh Saxena**  
B.Tech Computer Science, BITS Pilani (2027)

[LinkedIn](https://www.linkedin.com/)
[GitHub](https://github.com/)

---

