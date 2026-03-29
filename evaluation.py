"""
evaluation.py
Computes regression metrics (MSE, RMSE, R², MAE) for each model and
produces a bar-chart comparison saved to results/model_performance.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import predict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    All metrics operate on log-scale predictions/targets to stay consistent
    with training.  RMSE here is therefore RMSLE.
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_all(trained_models: dict, X_val: np.ndarray,
                 y_val: np.ndarray) -> dict:
    """
    Parameters
    ----------
    trained_models : {name: model}
    X_val, y_val   : validation features & log-scale targets

    Returns
    -------
    results : {name: {MSE, RMSE, MAE, R2}}
    """
    results = {}
    for name, model in trained_models.items():
        y_pred = predict(model, X_val)
        results[name] = compute_metrics(y_val, y_pred)
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics(results: dict):
    header = f"{'Model':<22} {'MSE':>10} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        print(f"{name:<22} {m['MSE']:>10.4f} {m['RMSE']:>8.4f} "
              f"{m['MAE']:>8.4f} {m['R2']:>8.4f}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, save_dir: str = "results"):
    """
    Four side-by-side bar charts (one per metric).
    Saves to <save_dir>/model_performance.png and also shows interactively.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(results.keys())
    metrics     = ["MSE", "RMSE", "MAE", "R2"]
    titles      = ["Mean Squared Error (↓)", "RMSE (↓)", "MAE (↓)", "R² Score (↑)"]
    colours     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    for ax, metric, title, colour in zip(axes, metrics, titles, colours):
        values = [results[m][metric] for m in model_names]
        bars   = ax.bar(model_names, values, color=colour, edgecolor="white",
                        linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "model_performance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")
    plt.show()


def plot_predictions_vs_actual(trained_models: dict, X_val: np.ndarray,
                                y_val: np.ndarray, save_dir: str = "results"):
    """Scatter plot of predicted vs. actual (log-scale) for each model."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = predict(model, X_val)
        ax.scatter(y_val, y_pred, alpha=0.4, s=15, color="#4C72B0")
        lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("Actual (log scale)")
        ax.set_ylabel("Predicted (log scale)")
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle("Predicted vs Actual Sale Price (log scale)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "predictions_vs_actual.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out_path}")
    plt.show()
