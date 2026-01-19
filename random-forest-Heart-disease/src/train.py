# src/train.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier



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


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
        out["PR_AUC"] = float(average_precision_score(y_true, y_proba))
    return out


def make_model(cfg: dict) -> RandomForestClassifier:
    m = cfg.get("model", {})
    return RandomForestClassifier(
        random_state=cfg["data"]["random_state"],
        max_depth=m.get("max_depth", None),
        n_estimators=m.get("n_estimators", 1),
        min_samples_leaf=m.get("min_samples_leaf", 2),
        oob_score=m.get("oob_score", 2),
        class_weight=m.get("class_weight", None),
    )


def choose_threshold_from_pr(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    strategy: dict,
) -> tuple[float, dict]:
    """
    precision, recall have length n_thresholds + 1
    thresholds has length n_thresholds
    We align by using precision[:-1], recall[:-1] vs thresholds.
    """
    p = precision[:-1]
    r = recall[:-1]
    t = thresholds

    if len(t) == 0:
        return 0.5, {"type": "fixed", "value": 0.5}

    stype = (strategy or {}).get("strategy", "f1_max")

    # Strategy: maximize F1 : Use when You want balanced performance
    if stype == "f1_max":
        f1 = (2 * p * r) / (p + r + 1e-12) # compute F1 at every threshold
        idx = int(np.argmax(f1)) # Here get the index of the biggest F1 value
        return float(t[idx]), {"type": "f1_max", "f1": float(f1[idx])} # Return the threshold where F1 is highest

    # Strategy: meet minimum precision, maximize recall
    if stype == "min_precision": # USE WHEN False positives are costly (spam, finance, alerts)
        min_p = float(strategy.get("min_precision", 0.80)) 
        ok = np.where(p >= min_p)[0] # thresholds where precision >= min_precision
        if len(ok) == 0: # If none satisfy to the condition
            idx = int(np.argmax(p)) # take the index of the higher precision
            return float(t[idx]), {"type": "min_precision_unmet", "min_precision": min_p} # Return the threshold with best precision possible
        idx = int(ok[np.argmax(r[ok])]) # If my condition is satisfy, among them choose the one with highest recall.
        return float(t[idx]), {"type": "min_precision", "min_precision": min_p}

    # Strategy: meet minimum recall, maximize precision
    if stype == "min_recall": # USE WHEN Missing positives is dangerous (medical, fraud, safety)
        min_r = float(strategy.get("min_recall", 0.80))
        ok = np.where(r >= min_r)[0]
        if len(ok) == 0:
            idx = int(np.argmax(r))
            return float(t[idx]), {"type": "min_recall_unmet", "min_recall": min_r}
        idx = int(ok[np.argmax(p[ok])])
        return float(t[idx]), {"type": "min_recall", "min_recall": min_r}

    # Strategy: keep precision & recall in ranges, maximize F1 within feasible region
    if stype == "range":
        pr_lo, pr_hi = strategy.get("precision_range", [0.0, 1.0])
        rc_lo, rc_hi = strategy.get("recall_range", [0.0, 1.0])
        pr_lo, pr_hi = float(pr_lo), float(pr_hi)
        rc_lo, rc_hi = float(rc_lo), float(rc_hi)

        ok = np.where((p >= pr_lo) & (p <= pr_hi) & (r >= rc_lo) & (r <= rc_hi))[0]
        if len(ok) == 0:
            # fallback: maximize F1 overall
            f1 = (2 * p * r) / (p + r + 1e-12)
            idx = int(np.argmax(f1))
            return float(t[idx]), {
                "type": "range_unmet_fallback_f1",
                "precision_range": [pr_lo, pr_hi],
                "recall_range": [rc_lo, rc_hi],
            }
        f1_ok = (2 * p[ok] * r[ok]) / (p[ok] + r[ok] + 1e-12)
        idx = int(ok[np.argmax(f1_ok)])
        return float(t[idx]), {
            "type": "range",
            "precision_range": [pr_lo, pr_hi],
            "recall_range": [rc_lo, rc_hi],
        }

    # fallback
    return 0.5, {"type": "fixed", "value": 0.5}


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

    # Threshold selection config (used for cv or VAL)
    thsel = (cfg.get("evaluation", {}) or {}).get("threshold_selection", {}) or {}
    thsel_method = thsel.get("method", "cv").lower()

    if mode == "cv":
        k = int(cfg["training"].get("cv_folds", 5))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["data"]["random_state"])

        fold_metrics = []
        cv_proba = np.full(shape=(len(y_train),), fill_value=np.nan, dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y_train), y_train), start=1):
            X_tr = X_train[tr_idx] if sparse.issparse(X_train) else X_train[tr_idx, :]
            X_va = X_train[va_idx] if sparse.issparse(X_train) else X_train[va_idx, :]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model_fold = make_model(cfg)
            model_fold.fit(X_tr, y_tr)

            y_va_pred = model_fold.predict(X_va)

            y_va_proba = None
            if len(np.unique(y_train)) == 2:
                y_va_proba = model_fold.predict_proba(X_va)[:, 1]
                cv_proba[va_idx] = y_va_proba

            m = classification_metrics(y_va, y_va_pred, y_va_proba)
            m["fold"] = fold
            fold_metrics.append(m)

        df = pd.DataFrame(fold_metrics)
        train_report["cv"] = {"folds": k}
        for col in df.columns:
            if col == "fold":
                continue
            train_report["cv"][f"{col}_mean"] = float(df[col].mean())
            train_report["cv"][f"{col}_std"] = float(df[col].std(ddof=1))

        # Choose threshold using cvcross validation) probs :
        if thsel_method == "cv":
            if np.isnan(cv_proba).any():
                raise ValueError("cv probabilities contain NaN. Is your task binary classification?")
            precision, recall, thresholds = precision_recall_curve(y_train, cv_proba)
            chosen_th, chosen_meta = choose_threshold_from_pr(precision, recall, thresholds, thsel)
            train_report["threshold_selection"] = {
                "method": "cv",
                "chosen_threshold": float(chosen_th),
                "details": chosen_meta,
            }

        # Fit final model on all training data
        final_model = make_model(cfg)
        final_model.fit(X_train, y_train)

    elif mode == "val":
        # threshold selection should be done on X_val/y_val (not test)
        X_val = load_matrix(os.path.join(processed_dir, "X_val"))
        y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

        final_model = make_model(cfg)
        final_model.fit(X_train, y_train)

        if thsel_method == "val":
            y_val_proba = final_model.predict_proba(X_val)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
            chosen_th, chosen_meta = choose_threshold_from_pr(precision, recall, thresholds, thsel)
            train_report["threshold_selection"] = {
                "method": "val",
                "chosen_threshold": float(chosen_th),
                "details": chosen_meta,
            }

        # still log val metrics at default 0.5 or chosen threshold if you want
        y_val_pred = final_model.predict(X_val)
        y_val_proba = final_model.predict_proba(X_val)[:, 1]
        train_report["val"] = classification_metrics(y_val, y_val_pred, y_val_proba)

    else:
        raise ValueError("training.mode must be 'cv' or 'val'")

    model_path = os.path.join(models_dir, cfg["model"]["name"])
    joblib.dump(final_model, model_path)
    train_report["model_path"] = model_path

    with open(os.path.join(reports_dir, "training_report.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(train_report, f, sort_keys=False)

    return train_report


if __name__ == "__main__":
    report = train()
    print("✅ Decision Tree classification training done.")
    print(report)
