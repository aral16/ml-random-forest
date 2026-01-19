# Online Purchase Intent Prediction using Random Forest

## Problem
E-commerce platforms aim to identify browsing sessions that are likely to convert into purchases.  
This project predicts whether an online shopping session will generate revenue using behavioral features.

## Dataset
Source: UCI Online Shoppers Purchasing Intention  

- 12,330 sessions  
- Mix of behavioral and categorical features (pages visited, bounce rate, month, visitor type, traffic type, etc.)  
- Target:
  - Revenue = True → purchase occurred
  - Revenue = False → no purchase  

The dataset is highly imbalanced (~15% buyers).

## Why Random Forest?
A single shallow Decision Tree captures interpretable rules but struggles with recall stability and class imbalance.  
Random Forest improves prediction by:

- Training many trees on bootstrapped samples  
- Randomizing feature selection at each split  
- Aggregating predictions to reduce variance  
- Providing more robust probability estimates  

This makes it ideal for behavioral classification on tabular data.

## Model
RandomForestClassifier  

- n_estimators = 200  
- max_depth = 6  
- min_samples_leaf = 5  
- class_weight = balanced  
- oob_score = True  
- Stratified 5-fold Cross-Validation  

## Threshold Strategy
Threshold selected using CV out-of-fold predictions (no test leakage).

Goal:
- Maintain Recall ≥ 80% (detect most buyers)
- Maximize Precision under that constraint

Chosen threshold: 0.551


---

## Cross-Validation Performance

| Metric | Mean |
|--------|------|
Accuracy | 0.875 |
Precision | 0.564 |
Recall | 0.832 |
F1 | 0.673 |
ROC-AUC | 0.925 |
PR-AUC | 0.721 |

---

## Test Performance (Frozen Threshold)

| Metric | Value |
|--------|-------|
Accuracy | 0.872 |
Precision | 0.564 |
Recall | 0.752 |
F1 | 0.644 |
ROC-AUC | 0.915 |
PR-AUC | 0.680 |

Confusion Matrix:

[[2794, 333],
[142, 430]]


---

## Comparison vs Decision Tree

| Metric | Decision Tree | Random Forest |
|--------|---------------|---------------|
Accuracy | 0.867 | **0.872** |
Precision | 0.550 | **0.564** |
Recall | 0.764 | **0.752** |
ROC-AUC | 0.899 | **0.915** |
PR-AUC | 0.627 | **0.680** |

Random Forest provides stronger ranking performance and more stable recall across folds.

---

## Key Insights
- Random Forest captures complex behavioral interactions missed by a shallow tree.
- Probability estimates are smoother and more reliable.
- PR-AUC improvement shows better detection of rare purchasing sessions.
- Ensemble learning improves generalization on imbalanced tabular data.

## Limitations
- No temporal sequence modeling
- Dataset imbalance still impacts precision
- No direct revenue/profit optimization

## Conclusion
Random Forest significantly improves behavioral purchase prediction by reducing variance and increasing recall stability across sessions.  
This project highlights how ensemble models outperform interpretable trees in real-world e-commerce classification tasks.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
