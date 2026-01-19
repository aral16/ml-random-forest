# src/train.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_matrix(path_base: str):
    if path_base.endswith(".npz"):
        return sparse.load_npz(path_base)
    if path_base.endswith(".npy"):
        return np.load(path_base, allow_pickle=False)
    if os.path.exists(path_base + ".npz"):
        return sparse.load_npz(path_base + ".npz")
    return np.load(path_base + ".npy", allow_pickle=False)


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def make_model(cfg: dict) -> RandomForestRegressor:
    m = cfg.get("model", {})
    return RandomForestRegressor(
        random_state=cfg["data"]["random_state"],
        max_depth=m.get("max_depth", None),
        n_estimators=m.get("n_estimators", 1),
        min_samples_leaf=m.get("min_samples_leaf", 2),
        n_jobs=m.get("n_jobs", -1),
    )

def train(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)

    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    mode = cfg["training"].get("mode", "cv").lower()

    X_train = load_matrix(os.path.join(processed_dir, "X_train"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))

    train_report = {"training_mode": mode, "model": cfg.get("model", {})}

    if mode == "cv":
        k = int(cfg["training"].get("cv_folds", 5))
        kf = KFold(n_splits=k, shuffle=True, random_state=cfg["data"]["random_state"])

        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(y_train), start=1):
            X_tr = X_train[tr_idx] if sparse.issparse(X_train) else X_train[tr_idx, :]
            X_va = X_train[va_idx] if sparse.issparse(X_train) else X_train[va_idx, :]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model_fold = make_model(cfg)
            model_fold.fit(X_tr, y_tr)
            y_va_pred = model_fold.predict(X_va)

            m = regression_metrics(y_va, y_va_pred)
            m["fold"] = fold
            fold_metrics.append(m)

        df = pd.DataFrame(fold_metrics)
        train_report["cv"] = {
            "folds": k,
            "MAE_mean": float(df["MAE"].mean()),
            "MAE_std": float(df["MAE"].std(ddof=1)),
            "RMSE_mean": float(df["RMSE"].mean()),
            "RMSE_std": float(df["RMSE"].std(ddof=1)),
            "R2_mean": float(df["R2"].mean()),
            "R2_std": float(df["R2"].std(ddof=1)),
        }

        # Fit final model on all training data
        final_model = make_model(cfg)
        final_model.fit(X_train, y_train)

    elif mode == "val":
        X_val = load_matrix(os.path.join(processed_dir, "X_val"))
        y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

        final_model = make_model(cfg)
        final_model.fit(X_train, y_train)
        y_val_pred = final_model.predict(X_val)
        train_report["val"] = regression_metrics(y_val, y_val_pred)

    else:
        raise ValueError("training.mode must be 'cv' or 'val'")

    # Save model
    model_path = os.path.join(models_dir, "model.joblib")
    joblib.dump(final_model, model_path)
    train_report["model_path"] = model_path

    # Save training report
    with open(os.path.join(reports_dir, "training_report.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(train_report, f, sort_keys=False)

    return train_report


if __name__ == "__main__":
    report = train()
    print("✅ Decision Tree training done.")
    print(report)
