# src/preprocess.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from typing import Optional, Tuple
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



@dataclass
class PreprocessArtifacts:
    X_train_path: str
    y_train_path: str
    X_val_path: Optional[str]
    y_val_path: Optional[str]
    X_test_path: str
    y_test_path: str
    feature_names: list[str]
    preprocessor_path: str 


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_matrix(path_base: str, X):
    """
    Saves X as sparse .npz if sparse, else dense .npy.
    Returns actual saved path.
    """
    if sparse.issparse(X):
        out = path_base + ".npz"
        sparse.save_npz(out, X)
        return out
    else:
        out = path_base + ".npy"
        np.save(out, X)
        return out

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: names unavailable (rare). Still works, but interpretation is harder.
        return []
    

def preprocess(config_path: str = "config.yaml") -> PreprocessArtifacts:
    cfg = load_config(config_path)

    raw_dir = cfg["paths"].get("raw_dir", "data/raw")
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    ensure_dir(processed_dir)
    ensure_dir(models_dir)

    # ---- 1) Load data file (raw input) ----
    train_path = os.path.join(raw_dir, "online_shoppers_intention.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}. Put Kaggle online_shoppers_intention.csv into data/raw/")

    df = pd.read_csv(train_path)


    # ---- 2) Selecting the target ol (y)----
    target_col = cfg["data"].get("target_col")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train.csv")

    y = df[target_col].to_numpy() # Here, I'm  NOT log transform the target column value  CAUSE WE SHOULDN't for decision-tree model.

    X = df.drop(columns=[target_col])
    
    # ---- 3) Split TEST first ----
    X_train_full, X_test_df, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    mode = cfg["training"].get("mode", "cv").lower()
    X_val_df = None
    y_val = None

    # 4) Optional validation split (from train only)
    if mode == "val":
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=cfg["training"]["val_size"],
            random_state=cfg["data"]["random_state"],
        )
    else:
        X_train_df, y_train = X_train_full, y_train_full

    # 5) Column types (based on TRAIN only)
    num_cols = X_train_df.select_dtypes(include=["number"]).columns.tolist() # Here i'm finding which colum are numeric
    cat_cols = X_train_df.select_dtypes(exclude=["number"]).columns.tolist() # Here i'm finding which colum are categorical

    # 6) Preprocessing pipelines
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # Here, we're imputing missing values with median. So basically, the missing value (NA) will be replace by the median value
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),# Here, we're filling missing values with the most common category
        ("onehot", OneHotEncoder(handle_unknown="ignore")), # Here, we're One-hot encoding categories into binary columns
    ])

    # Now, let's combine the numerical and categorical transformation
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols), # Here, we're applying numeric pipeline only to numeric columns.
            ("cat", categorical_pipe, cat_cols), # Here, we're applying categorical pipeline only to categorical columns.
        ],
        remainder="drop" # Here we're droping any other columns not explicitly listed like categorical or numeric.
    )

    # 7) Fit on TRAIN only, transform all splits (prevents leakage)
    X_train = preprocessor.fit_transform(X_train_df) # During the transform step, Any missing numeric entry will be replaced with stored median.
    X_test = preprocessor.transform(X_test_df)# During the transform step, Any missing numeric entry will be replaced with stored median.
    X_val = preprocessor.transform(X_val_df) if X_val_df is not None else None # During the transform step, Any missing numeric entry will be replaced with stored median.

    # 8) Save artifacts
    X_train_path = save_matrix(os.path.join(processed_dir, "X_train"), X_train)
    y_train_path = os.path.join(processed_dir, "y_train.npy")
    np.save(y_train_path, y_train)

    X_test_path = save_matrix(os.path.join(processed_dir, "X_test"), X_test)
    y_test_path = os.path.join(processed_dir, "y_test.npy")
    np.save(y_test_path, y_test)

    X_val_path = None
    y_val_path = None
    if X_val is not None:
        X_val_path = save_matrix(os.path.join(processed_dir, "X_val"), X_val)
        y_val_path = os.path.join(processed_dir, "y_val.npy")
        np.save(y_val_path, y_val)

    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib") # If you deploy the model later, you must:transform new data exactly the same way,with the same imputation medians,same scaler means/stds and same one-hot category map. Saving the preprocessor guarantees that
    joblib.dump(preprocessor, preprocessor_path)

    feature_names = get_feature_names(preprocessor)  # This creates a list of transformed feature names after one-hot encoding.
    joblib.dump(feature_names, os.path.join(processed_dir, "feature_names.joblib")) # And this save those features name

    return PreprocessArtifacts(
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_val_path=X_val_path,
        y_val_path=y_val_path,
        X_test_path=X_test_path,
        y_test_path=y_test_path,
        feature_names=feature_names,
        preprocessor_path=preprocessor_path
    )


if __name__ == "__main__":
    artifacts = preprocess("config.yaml")
    print("✅ Preprocessing done.")
    print(artifacts)
