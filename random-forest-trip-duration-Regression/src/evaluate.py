# src/evaluate.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    # allow passing exact filename too
    if path_base.endswith(".npz"):
        return sparse.load_npz(path_base)
    if path_base.endswith(".npy"):
        return np.load(path_base, allow_pickle=False)
    raise FileNotFoundError(f"Could not find {path_base}.npz or {path_base}.npy")


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}



def evaluate(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)

    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    ensure_dir(reports_dir)
    ensure_dir(figures_dir)

    log_target = bool(cfg.get("data", {}).get("log_target", False)) # Here it's to detect whether it's trained on log-transformed target or not.

    # Load test data
    X_test = load_matrix(os.path.join(processed_dir, "X_test"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    # Load model
    model_name = cfg["model"]["name"]
    model = joblib.load(os.path.join(models_dir, model_name)) # Here I'm loading the model, that have the model name define in the config file.

    # Predict and metrics
    y_pred = model.predict(X_test) # Predict (in whatever space the model was trained on: log transform or real values)
    
    report = {}
    if log_target:
        # y_test and y_pred are in log1p space -> convert back to original scale
        y_test_orig = np.expm1(y_test) # So here we're are removing the log transformation and putting the value back to their initial price without log transformation. expm1 mean : exp(z) - 1.
        y_pred_orig = np.expm1(y_pred)# Here also, we're are removing the log transformation and putting the value back to their initial price without log transformation

        # Metrics on ORIGINAL scale (most interpretable)
        report["test"] = regression_metrics(y_test_orig, y_pred_orig)

        # Optional: also report metrics in LOG space (useful for debugging)
        #report["test_log_space"] = regression_metrics(y_test, y_pred)

        # Residuals on original scale
        residuals = y_test_orig - y_pred_orig # Residuals tell you how your model is wrong, not just how much. so. it's elle you how far each prediction is from reality and also in which direction the model is wrong
        y_for_plot = y_pred_orig
        plot_suffix = "_orig"
        resid_title = "Residual Distribution (original scale)"
        scatter_title = "Residuals vs Predictions (original scale)"
        x_label = "Predicted (original scale)"
        y_label = "Residual (true - pred) (original scale)"
    else:
        # Normal case: everything already in original space
        report["test"] = regression_metrics(y_test, y_pred)

        residuals = y_test - y_pred
        y_for_plot = y_pred
        plot_suffix = ""
        resid_title = "Residual Distribution (y_true - y_pred)"
        scatter_title = "Residuals vs Predictions"
        x_label = "Predicted"
        y_label = "Residual (true - pred)"

    # Plots
    if cfg.get("evaluation", {}).get("save_figures", True): # Here, we're Plotting (residual diagnostics)
        # Residual histogram
        plt.figure()
        plt.hist(residuals, bins=40)
        plt.title(resid_title)
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"residual_hist{plot_suffix}.png"), dpi=160)
        plt.close()

        # Residual vs predicted
        plt.figure()
        plt.scatter(y_for_plot, residuals, s=8)
        plt.axhline(0)
        plt.title(scatter_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"residuals_vs_pred{plot_suffix}.png"), dpi=160)
        plt.close()

    return report



def merge_metrics(training_report: dict, eval_report: dict, out_path: str):
    combined = {"training": training_report, **eval_report}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(combined, f, sort_keys=False)


if __name__ == "__main__":
    # Load training report if present (optional: you can save it from train.py as well)
    # Simple approach: re-run train() and capture output manually OR store it yourself.
    # Here we try to load a cached training report if you saved one.
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
