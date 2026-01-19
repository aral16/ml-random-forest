# Random Forest Regression — NYC Taxi Trip Duration

## Problem
Predict the duration of taxi trips in New York City using spatio-temporal and trip-related features.

## Dataset
Kaggle — **NYC Taxi Trip Duration**

Target:
- `trip_duration` (seconds)

Features include:
- Pickup and dropoff coordinates  
- Vendor  
- Passenger count  
- Datetime of trip  

This dataset is known to be challenging due to strong spatial effects, traffic variability, and extreme outliers.

---

## ML Task
Random Forest Regression

---

## Approach
1. Train/Validation/Test split  
2. Minimal preprocessing (no scaling required for trees)  
3. RandomForestRegressor trained with:
   - 300 trees
   - max_depth = 12
   - min_samples_leaf = 5  
4. Evaluation on validation and test sets  
5. Residual diagnostics

---

## Validation Results

| Metric | Value |
|--------|-------|
MAE | 471.7 seconds |
RMSE | 5862.2 seconds |
R² | -0.038 |

---

## Test Results

| Metric | Value |
|--------|-------|
MAE | 477.2 seconds |
RMSE | 3375.5 seconds |
R² | -0.116 |

Negative R² indicates the model performs worse than predicting the mean trip duration.

---

## Residual Diagnostics

### Residual Distribution
- Extremely wide error spread  
- Heavy skew  
- Presence of large outliers  

### Residuals vs Predictions
- Strong downward trend → systematic overprediction for long trips  
- Large variance for higher predicted durations  
- Model instability across spatial ranges  

These patterns show the forest failed to capture core nonlinear relationships driving trip time.

---

## Why the Model Failed
The poor performance is mainly due to:

- Lack of engineered distance features  
- Raw latitude/longitude not transformed into meaningful spatial signals  
- Heavy target skew and extreme trip outliers  
- Traffic/time interactions not captured by simple features  

Random Forest alone is not sufficient without strong feature engineering for geospatial regression problems.

---

## Key Takeaways
- Tree ensembles still require proper features for spatial problems.
- Raw coordinates are not informative enough for regression.
- Outlier-heavy targets can break tree models.
- This dataset strongly benefits from feature engineering and boosting methods.

---

## Conclusion
This experiment highlights the limitations of Random Forest regression when applied to complex geospatial-temporal datasets without advanced feature engineering. It provides a valuable baseline illustrating why more specialized techniques (distance computation, clustering, gradient boosting) are necessary for accurate trip duration prediction.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
