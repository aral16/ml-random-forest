# 🌳 Machine Learning with Random Forest

This repository showcases three end-to-end machine learning projects using **Random Forest** across different domains:

- 🏥 Healthcare classification  
- 🛒 E-commerce behavioral classification  
- 🚕 Geospatial regression  

The goal is to demonstrate how Random Forest behaves under varied data structures, objectives, and constraints.

---

## 📂 Projects Overview

| Project | Task | Dataset | Objective |
|---------|------|---------|-----------|
Heart Disease | Classification | UCI Cleveland | Predict disease presence |
Purchase Intent | Classification | UCI Online Shoppers | Predict session conversion |
Taxi Duration | Regression | Kaggle NYC Taxi | Predict trip duration |

---

# 🏥 Project 1 — Heart Disease Classification

Predict whether a patient has heart disease using structured clinical features.

**Highlights**
- Binary classification
- Strong non-linear interactions
- Class imbalance handled with `class_weight=balanced`
- Threshold tuned using CV out-of-fold probabilities

**Outcome**
Random Forest significantly improved recall, ROC-AUC and PR-AUC compared to a single Decision Tree.

---

# 🛒 Project 2 — Online Purchase Intent

Predict whether an online shopping session results in revenue.

**Highlights**
- Behavioral + categorical data
- Highly imbalanced (~15% buyers)
- Forest captures complex browsing patterns
- Threshold tuned to prioritize recall (marketing use case)

**Outcome**
Random Forest improved stability, recall and PR-AUC over the shallow tree baseline.

---

# 🚕 Project 3 — Taxi Trip Duration (Regression)

Predict NYC taxi trip duration using spatial and temporal features.

**Highlights**
- Continuous regression target
- Geospatial coordinates
- High noise + extreme outliers
- No distance feature engineered (intentional baseline)

**Outcome**
Random Forest fails (negative R²), illustrating limitations of tree ensembles when critical spatial features are missing.

This project demonstrates an important ML lesson:
> Even strong ensemble models require meaningful feature engineering for spatial regression tasks.

---

## Key Takeaways Across Projects

- Random Forest excels in tabular classification.
- Robust to non-linear interactions and noisy features.
- Provides stable feature importance.
- Handles imbalance better than single trees.
- Regression performance heavily depends on feature quality.
- Poorly engineered spatial features can break forests.

---

## Next Steps (Future Work)

- Improve Taxi model with distance features (Haversine).
- Compare RF vs Gradient Boosting.
- Explore XGBoost / LightGBM for performance gains.

---

## How to Run Any Project

Each subfolder contains its own pipeline:

```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
