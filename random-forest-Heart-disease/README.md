# Heart Disease Prediction using Decision Trees

## Problem
Early identification of heart disease is critical in clinical decision-making.  
The objective of this project is to predict whether a patient presents heart disease using structured clinical features.

## Dataset
Source: UCI Heart Disease (Cleveland)  
- 303 patients  
- 13 clinical features (demographics, ECG, exercise, blood markers)  
- Original labels (0–4) converted to binary:
  - 0 → No disease
  - 1–4 → Disease present

## Why Decision Trees?
Decision Trees are well suited for:
- Non-linear clinical relationships
- Mixed feature types
- Human-interpretable rule extraction

They mimic medical reasoning such as:
> IF chest pain type AND thal level AND age → disease risk

## Model
DecisionTreeClassifier  
- max_depth = 8  
- min_samples_leaf = 10  
- Stratified 5-fold Cross-Validation

## Threshold Strategy
Threshold was selected using CV out-of-fold predictions (no test leakage).

Goal:
- Maintain recall ≥ 80%
- Maximize precision under that constraint

Chosen threshold: 0.5833


## Cross-Validation Performance

| Metric | Mean |
|--------|------|
Accuracy | 0.778 |
Precision | 0.840 |
Recall | 0.661 |
F1 | 0.731 |
ROC-AUC | 0.854 |
PR-AUC | 0.825 |

## Test Performance (Frozen Threshold)

| Metric | Value |
|--------|-------|
Accuracy | 0.736 |
Precision | 0.781 |
Recall | 0.595 |
F1 | 0.676 |
ROC-AUC | 0.836 |
PR-AUC | 0.772 |

Confusion Matrix:

[[42, 7],
[17, 25]]

---

## Key Insights
- The model captures meaningful non-linear interactions between clinical variables.
- Threshold tuning allowed prioritization of reliable disease predictions.
- Slight performance drop from CV to test indicates mild overfitting, expected for trees.

## Limitations
- Small dataset size
- No external validation cohort
- Model is for educational/analytical use only

## Conclusion
Decision Trees provide interpretable and clinically meaningful predictions.  
This project demonstrates how classification models can be tuned using precision-recall tradeoffs and deployed responsibly without threshold leakage.

---


## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
