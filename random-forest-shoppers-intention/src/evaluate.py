# src/evaluate.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_matrix(path_base: str):
    if os.path.exists(path_base + ".npz"):
        return sparse.load_npz(path_base + ".npz")
    if os.path.exists(path_base + ".npy"):
        return np.load(path_base + ".npy", allow_pickle=False)
    if path_base.endswith(".npz"):
        return sparse.load_npz(path_base)
    if path_base.endswith(".npy"):
        return np.load(path_base, allow_pickle=False)
    raise FileNotFoundError(f"Could not find {path_base}.npz or {path_base}.npy")


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["PR_AUC"] = float(average_precision_score(y_true, y_proba))
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
    return out


def evaluate(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)

    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    ensure_dir(reports_dir)
    ensure_dir(figures_dir)

    # Load training report to get frozen threshold
    training_report_path = os.path.join(reports_dir, "training_report.yaml")
    if not os.path.exists(training_report_path):
        raise FileNotFoundError("Missing reports/training_report.yaml (needed to get chosen threshold).")

    with open(training_report_path, "r", encoding="utf-8") as f:
        training_report = yaml.safe_load(f) or {}

    th_info = training_report.get("threshold_selection", {})
    threshold = th_info.get("chosen_threshold", 0.5)
    threshold_method = th_info.get("method", "none")

    # Load test data
    X_test = load_matrix(os.path.join(processed_dir, "X_test"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    # Load model
    model_name = cfg["model"]["name"]
    model = joblib.load(os.path.join(models_dir, model_name))

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply frozen threshold
    y_pred = (y_proba >= float(threshold)).astype(int)

    report = {
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "test": classification_metrics(y_test, y_pred, y_proba),
    }

    # Diagnostics plots (NOT used to choose threshold)
    if cfg.get("evaluation", {}).get("save_figures", True):
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # Precision/Recall vs threshold
        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, precision[:-1], label="Precision")
        plt.plot(thresholds, recall[:-1], label="Recall")
        plt.axvline(float(threshold), linestyle="--", label=f"Frozen threshold = {float(threshold):.2f}")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision & Recall vs Threshold (test diagnostic)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "precision_recall_vs_threshold_test.png"), dpi=160)
        plt.close()

        # PR curve
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (test diagnostic)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "pr_curve_test.png"), dpi=160)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (test diagnostic)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "roc_curve_test.png"), dpi=160)
        plt.close()

    return report


def merge_metrics(training_report: dict, eval_report: dict, out_path: str):
    combined = {"training": training_report, **eval_report}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(combined, f, sort_keys=False)


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dir(reports_dir)

    training_report_path = os.path.join(reports_dir, "training_report.yaml")
    training_report = {}
    if os.path.exists(training_report_path):
        with open(training_report_path, "r", encoding="utf-8") as f:
            training_report = yaml.safe_load(f) or {}

    eval_report = evaluate("config.yaml")

    metrics_path = os.path.join(reports_dir, "metrics.yaml")
    merge_metrics(training_report, eval_report, metrics_path)

    print("✅ Evaluation done.")
    print("Saved metrics to:", metrics_path)
    print(eval_report)
