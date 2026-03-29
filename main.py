"""
main.py
Entry point for the Housing Price Prediction & Affordability Analysis project.

Pipeline
--------
1. Load & preprocess data          (preprocessing.py)
2. Visualise target distribution   (plot_target_dist_classification.py)
3. Visualise correlation heatmap   (correlation_heatmap_plot.py)
4. Train all models                (models.py)
5. Evaluate & compare models       (evaluation.py)
6. Affordability analysis          (affordability.py)
"""

import numpy as np

# ── 1. Preprocessing ────────────────────────────────────────────────────────
print("=" * 60)
print("  Housing Price Prediction & Affordability Analysis")
print("=" * 60)

print("\n[1/6] Preprocessing data …")
from preprocessing import prepare_data

X_train, X_val, y_train, y_val, feature_names, scaler = prepare_data(
    test_size=0.2, random_state=42
)
print(f"  Train samples : {X_train.shape[0]}")
print(f"  Val   samples : {X_val.shape[0]}")
print(f"  Features      : {X_train.shape[1]}")

# ── 2. Target distribution plot ─────────────────────────────────────────────
print("\n[2/6] Plotting target distribution …")
from plot_target_dist_classification import plot_target_distribution_classification
plot_target_distribution_classification()

# ── 3. Correlation heatmap ───────────────────────────────────────────────────
print("\n[3/6] Plotting correlation heatmap (top 12 features) …")
from correlation_heatmap_plot import plot_corr_heatmap_top_k_features
plot_corr_heatmap_top_k_features(k=12)

# ── 4. Train models ──────────────────────────────────────────────────────────
print("\n[4/6] Training models …")
from models import train_all
trained_models = train_all(X_train, y_train, X_val, y_val, verbose=True)

# ── 5. Evaluate & compare ────────────────────────────────────────────────────
print("\n[5/6] Evaluating models …")
from evaluation import (
    evaluate_all,
    print_metrics,
    plot_model_comparison,
    plot_predictions_vs_actual,
)

results = evaluate_all(trained_models, X_val, y_val)
print_metrics(results)
plot_model_comparison(results)
plot_predictions_vs_actual(trained_models, X_val, y_val)

# ── 6. Affordability analysis ────────────────────────────────────────────────
print("\n[6/6] Affordability analysis …")
from affordability import (
    print_affordability_summary,
    plot_affordability_distribution,
    plot_price_distribution_by_class,
)

# Use the best model (highest R²) for affordability analysis
best_name = max(results, key=lambda n: results[n]["R2"])
print(f"  Best model by R²: {best_name} (R² = {results[best_name]['R2']:.4f})")

from models import predict
y_val_pred = predict(trained_models[best_name], X_val)

print_affordability_summary(y_val_pred)
plot_affordability_distribution(y_val_pred, model_name=best_name)
plot_price_distribution_by_class(y_val_pred, model_name=best_name)

print("\nDone.")
