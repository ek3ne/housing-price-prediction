"""
affordability.py
Classifies predicted housing prices by affordability for five income classes
using the standard 28 % rule: a home is affordable when its price is ≤ 4× the
buyer's annual gross income (rough 30-year mortgage approximation).

Income brackets (Statistics Canada / US Census rough equivalents):
  Very Low  income : < $35 000 / yr  → affordable up to ~$140 000
  Low       income : $35 000–$60 000 → affordable up to ~$240 000
  Middle    income : $60 000–$100 000 → affordable up to ~$400 000
  Upper-Mid income : $100 000–$150 000 → affordable up to ~$600 000
  High      income : > $150 000       → affordable above  $600 000
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Income class definitions
# ---------------------------------------------------------------------------

INCOME_CLASSES = [
    {
        "label":        "Very Low\n(<$35k/yr)",
        "short":        "Very Low",
        "max_price":    140_000,
        "colour":       "#d62728",
    },
    {
        "label":        "Low\n($35k–$60k/yr)",
        "short":        "Low",
        "max_price":    240_000,
        "colour":       "#ff7f0e",
    },
    {
        "label":        "Middle\n($60k–$100k/yr)",
        "short":        "Middle",
        "max_price":    400_000,
        "colour":       "#2ca02c",
    },
    {
        "label":        "Upper-Middle\n($100k–$150k/yr)",
        "short":        "Upper-Middle",
        "max_price":    600_000,
        "colour":       "#1f77b4",
    },
    {
        "label":        "High\n(>$150k/yr)",
        "short":        "High",
        "max_price":    float("inf"),
        "colour":       "#9467bd",
    },
]


def classify_price(price: float) -> str:
    """Return the income-class label for a single predicted price."""
    for cls in INCOME_CLASSES:
        if price <= cls["max_price"]:
            return cls["short"]
    return INCOME_CLASSES[-1]["short"]


def classify_predictions(y_pred_log: np.ndarray) -> pd.DataFrame:
    """
    Parameters
    ----------
    y_pred_log : log1p-scale predictions (as returned by the models)

    Returns
    -------
    DataFrame with columns: PredictedPrice, AffordableFor
    """
    prices = np.expm1(y_pred_log)
    labels = [classify_price(p) for p in prices]
    return pd.DataFrame({"PredictedPrice": prices, "AffordableFor": labels})


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_affordability_distribution(y_pred_log: np.ndarray,
                                    model_name: str = "Best Model",
                                    save_dir: str = "results"):
    """
    Bar chart: how many predicted homes fall into each affordability tier.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = classify_predictions(y_pred_log)

    ordered_labels = [c["short"]   for c in INCOME_CLASSES]
    colours        = [c["colour"]  for c in INCOME_CLASSES]
    counts = df["AffordableFor"].value_counts().reindex(ordered_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ordered_labels, counts.values, color=colours, edgecolor="white",
                  linewidth=0.8)
    ax.set_title(f"Housing Affordability by Income Class\n({model_name})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Income Class", fontsize=11)
    ax.set_ylabel("Number of Homes", fontsize=11)

    for bar, val in zip(bars, counts.values):
        pct = val / counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{int(val)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    price_labels = [
        f"≤ ${c['max_price']:,}" if c["max_price"] != float("inf")
        else "> $600 000"
        for c in INCOME_CLASSES
    ]
    patches = [mpatches.Patch(color=c["colour"],
                              label=f"{c['short']}: {pl}")
               for c, pl in zip(INCOME_CLASSES, price_labels)]
    ax.legend(handles=patches, title="Affordable price range",
              loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "affordability_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out_path}")
    plt.show()


def plot_price_distribution_by_class(y_pred_log: np.ndarray,
                                     model_name: str = "Best Model",
                                     save_dir: str = "results"):
    """Box-plot of predicted prices grouped by affordability class."""
    os.makedirs(save_dir, exist_ok=True)
    df = classify_predictions(y_pred_log)

    ordered_labels = [c["short"] for c in INCOME_CLASSES]
    colours        = {c["short"]: c["colour"] for c in INCOME_CLASSES}

    groups = [df.loc[df["AffordableFor"] == lbl, "PredictedPrice"].values
              for lbl in ordered_labels]
    groups = [g for g in groups if len(g) > 0]
    active_labels = [lbl for lbl, g in zip(ordered_labels, groups) if len(g) > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, lbl in zip(bp["boxes"], active_labels):
        patch.set_facecolor(colours[lbl])
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(active_labels) + 1))
    ax.set_xticklabels(active_labels, fontsize=10)
    ax.set_title(f"Predicted Price Distribution by Income Class\n({model_name})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Income Class", fontsize=11)
    ax.set_ylabel("Predicted Sale Price ($)", fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    plt.tight_layout()
    out_path = os.path.join(save_dir, "price_distribution_by_class.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out_path}")
    plt.show()


def print_affordability_summary(y_pred_log: np.ndarray):
    df = classify_predictions(y_pred_log)
    total = len(df)
    print("\n" + "=" * 55)
    print(f"  Affordability Summary  (n={total} predictions)")
    print("=" * 55)
    print(f"  {'Income Class':<18} {'Count':>7}  {'Share':>7}  {'Median Price':>14}")
    print("-" * 55)
    for cls in INCOME_CLASSES:
        subset = df[df["AffordableFor"] == cls["short"]]
        if subset.empty:
            continue
        pct    = len(subset) / total * 100
        median = subset["PredictedPrice"].median()
        print(f"  {cls['short']:<18} {len(subset):>7}  {pct:>6.1f}%  ${median:>13,.0f}")
    print("=" * 55)
