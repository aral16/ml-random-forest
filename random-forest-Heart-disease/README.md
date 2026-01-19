# Heart Disease Prediction using Random Forest

## Problem
Early identification of heart disease is critical in clinical decision-making.  
This project predicts whether a patient presents heart disease using structured clinical features.

## Dataset
Source: UCI Heart Disease (Cleveland)

- 303 patients  
- 13 clinical features (demographics, ECG, exercise, blood markers)  
- Original labels (0–4) converted to binary:
  - 0 → No disease
  - 1–4 → Disease present

## Why Random Forest?
A single Decision Tree tends to overfit and is unstable.  
Random Forest improves reliability by:

- Training many trees on bootstrapped samples  
- Randomizing feature selection at each split  
- Aggregating predictions by majority vote  

This reduces variance and improves generalization while preserving interpretability via feature importance.

## Model
RandomForestClassifier

- n_estimators = 200  
- max_depth = 6  
- min_samples_leaf = 5  
- class_weight = balanced  
- oob_score = True  
- Stratified 5-fold Cross-Validation  

## Threshold Strategy
Threshold selected using **CV out-of-fold predictions** (no test leakage).

Goal:
- Maintain recall ≥ 80%
- Maximize precision under that constraint

Chosen threshold:0.4311


## Cross-Validation Performance

| Metric | Mean |
|--------|------|
Accuracy | 0.811 |
Precision | 0.813 |
Recall | 0.774 |
F1 | 0.791 |
ROC-AUC | 0.903 |
PR-AUC | 0.898 |

## Test Performance (Frozen Threshold)

| Metric | Value |
|--------|-------|
Accuracy | 0.780 |
Precision | 0.712 |
Recall | 0.881 |
F1 | 0.787 |
ROC-AUC | 0.928 |
PR-AUC | 0.926 |

Confusion Matrix:

[[34, 15],
[5, 37]]


## Comparison vs Single Decision Tree

| Metric | Decision Tree | Random Forest |
|--------|---------------|---------------|
Accuracy | 0.736 | **0.780** |
Precision | 0.667 | **0.712** |
Recall | 0.857 | **0.881** |
F1 | 0.750 | **0.787** |
ROC-AUC | 0.836 | **0.928** |
PR-AUC | 0.772 | **0.926** |

Random Forest clearly improves discrimination and stability while preserving high recall.

## Key Insights
- Ensemble learning dramatically improves model robustness on small clinical datasets.
- Feature importance becomes more stable across trees.
- Threshold tuning allows control over clinical tradeoffs (prioritizing sensitivity).
- Random Forest generalizes better than a single deep tree.

## Limitations
- Small dataset size
- No external validation cohort
- Predictions are not for medical use

## Conclusion
Random Forest significantly improves heart disease prediction by reducing overfitting and increasing stability across folds.  
This project demonstrates how ensemble models outperform single trees and how threshold tuning enables clinically meaningful decision boundaries.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
