# Housing Price Prediction & Affordability Analysis

Predicts housing sale prices using multiple machine learning models and classifies homes by affordability across different income brackets.

Built on the [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

---

## Motivation

Housing affordability is a growing concern for many people. This project uses machine learning to predict sale prices and identify which income classes can realistically afford homes in the dataset — providing data-driven insight into the housing market.

---

## Features

- Predicts housing prices from 80+ features (location, size, quality, amenities)
- Compares four machine learning models to find the best predictor
- Classifies predicted prices into five income-based affordability tiers
- Visualises model performance, correlations, and affordability distributions

---

## Models

| Model | Type |
|---|---|
| Linear Regression (Ridge) | Regularised linear baseline |
| Random Forest | Ensemble — handles non-linearity |
| Gradient Boosting | Strong ensemble baseline |
| Neural Network | Deep learning (Keras / sklearn MLP fallback) |

---

## Evaluation Metrics

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination

---

## Affordability Classification

Homes are classified using the **4× income rule** (price ≤ 4× annual gross income):

| Income Class | Annual Income | Affordable up to |
|---|---|---|
| Very Low | < $35 000 | ~ $140 000 |
| Low | $35 000 – $60 000 | ~ $240 000 |
| Middle | $60 000 – $100 000 | ~ $400 000 |
| Upper-Middle | $100 000 – $150 000 | ~ $600 000 |
| High | > $150 000 | > $600 000 |

---

## Technologies

- **Python 3** — primary language
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — preprocessing, Linear Regression, Random Forest, Gradient Boosting, MLP
- **TensorFlow / Keras** — neural network (optional)
- **Matplotlib** — visualisation

---

## Project Structure

```
housing-price-prediction/
│
├── Data/
│   ├── train.csv               # Training data (1 460 samples, 81 features)
│   ├── test.csv                # Test data
│   ├── sample_submission.csv   # Submission template
│   └── data_description.txt    # Feature descriptions
│
├── results/                    # Auto-generated charts
│   ├── model_performance.png
│   ├── predictions_vs_actual.png
│   ├── affordability_distribution.png
│   └── price_distribution_by_class.png
│
├── global_imports.py                   # Shared data loading
├── preprocessing.py                    # Missing value handling, encoding, scaling
├── models.py                           # Model training (all four models)
├── evaluation.py                       # Metrics and comparison charts
├── affordability.py                    # Income-class affordability analysis
├── correlation_heatmap_plot.py         # Correlation heatmap visualisation
├── plot_target_dist_classification.py  # Target distribution plot
└── main.py                             # Full pipeline entry point
```

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/ek3ne/housing-price-prediction.git
cd housing-price-prediction
```

2. Install dependencies:
```bash
pip install pandas scikit-learn matplotlib numpy
# Optional — for Keras neural network:
pip install tensorflow
```

3. Run the full pipeline:
```bash
python main.py
```

Charts are saved automatically to the `results/` folder.

---

## Results

- **Gradient Boosting** and **Random Forest** consistently outperform the linear baseline
- **Neural Network** captures complex non-linear patterns with sufficient data
- Key price drivers: `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `YearBuilt`
- Majority of homes in the Ames dataset are affordable for middle-income buyers

---

## Author

**Daniel** — University of Prince Edward Island, Computer Science
