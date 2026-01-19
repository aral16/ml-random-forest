# src/preprocess.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessArtifacts:
    X_train_path: str
    y_train_path: str
    X_val_path: str | None
    y_val_path: str | None
    X_test_path: str
    y_test_path: str
    preprocessor_path: str  # here: imputer.joblib


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def preprocess(config_path: str = "config.yaml") -> PreprocessArtifacts:
    cfg = load_config(config_path)

    raw_dir = cfg["paths"].get("raw_dir", "data/raw")
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    ensure_dir(processed_dir)
    ensure_dir(models_dir)

    # ---- 1) Load Cleveland file (raw input) ----
    cleveland_path = os.path.join(raw_dir, "processed.cleveland.data")
    if not os.path.exists(cleveland_path):
        raise FileNotFoundError(
            f"Missing {cleveland_path}. Put processed.cleveland.data into data/raw/"
        )

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]

    df = pd.read_csv(
        cleveland_path,
        header=None,
        names=columns,
        na_values="?",  # UCI uses "?" to mean missing
    )

    # ---- 2) Convert all columns to numeric (missing -> NaN) ----
    target_col = cfg["data"].get("target_col")
    
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 3) Binary target ----
    # Original target: 0 = no disease, 1-4 = disease severity
    df[target_col] = (df[target_col] > 0).astype(int) # So, here, converting everything > 0 to 1, so we could have only 2 class.

    # ---- 4) Split features/labels (keep NaNs for imputation) ----
    
    y = df[target_col].to_numpy(dtype=int)
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)

    # ---- 5) Split TEST first (never used for threshold selection / tuning) ----
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    mode = cfg["training"].get("mode", "cv").lower()

    # ---- 6) Optional validation split (from TRAIN only) ----
    if mode == "val":
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=cfg["training"]["val_size"],
            random_state=cfg["data"]["random_state"],
            stratify=y_train_full,
        )

        # ---- 7) Median imputation (fit ONLY on training split) ----
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_raw) # fit ONLY on training split
        X_val = imputer.transform(X_val_raw)
        X_test = imputer.transform(X_test)  # same imputer

    else:
        # mode == "cv" (or anything else treated as cv)
        # Fit imputer on full training set (still not using test)
        imputer = SimpleImputer(strategy="median") # So here,For each feature (column) independently, the median is compute and Replaces missing values in that column with that median.
        X_train = imputer.fit_transform(X_train_full) # fit ONLY on training split
        X_test = imputer.transform(X_test)

        y_train = y_train_full
        X_val = None
        y_val = None

    # ---- 8) Save arrays ----
    X_train_path = os.path.join(processed_dir, "X_train.npy")
    y_train_path = os.path.join(processed_dir, "y_train.npy")
    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)

    X_test_path = os.path.join(processed_dir, "X_test.npy")
    y_test_path = os.path.join(processed_dir, "y_test.npy")
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)

    X_val_path = None
    y_val_path = None
    if X_val is not None:
        X_val_path = os.path.join(processed_dir, "X_val.npy")
        y_val_path = os.path.join(processed_dir, "y_val.npy")
        np.save(X_val_path, X_val)
        np.save(y_val_path, y_val)

    # ---- 9) Save imputer for consistent inference ----
    preprocessor_path = os.path.join(models_dir, "imputer.joblib")
    joblib.dump(imputer, preprocessor_path)

    return PreprocessArtifacts(
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_val_path=X_val_path,
        y_val_path=y_val_path,
        X_test_path=X_test_path,
        y_test_path=y_test_path,
        preprocessor_path=preprocessor_path,
    )


if __name__ == "__main__":
    artifacts = preprocess("config.yaml")
    print("✅ Preprocessing done.")
    print(artifacts)
