"""
models.py
Trains and returns four regression models:
  1. Linear Regression  (interpretable baseline)
  2. Random Forest      (ensemble, handles non-linearity)
  3. Gradient Boosting  (strong ensemble baseline)
  4. Neural Network     (deep learning via Keras / sklearn fallback)

All models predict log1p(SalePrice); callers expm1 when needed.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _try_keras_nn(input_dim: int):
    """Build a Keras Sequential model; returns None if Keras unavailable."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return model
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual trainers
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train):
    """Ridge regression (regularised linear model to avoid overfitting)."""
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200,
                        random_state: int = 42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators: int = 300,
                             random_state: int = 42):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_neural_network(X_train, y_train, X_val, y_val,
                         epochs: int = 100, batch_size: int = 32):
    """
    Trains a Keras neural network when TensorFlow is installed.
    Falls back to sklearn's MLPRegressor otherwise.
    """
    keras_model = _try_keras_nn(X_train.shape[1])

    if keras_model is not None:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(factor=0.5, patience=7, verbose=0),
        ]
        keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return keras_model, "keras"

    # --- Fallback: sklearn MLP ---
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    return mlp, "sklearn"


# ---------------------------------------------------------------------------
# Unified predict helper (handles Keras vs sklearn)
# ---------------------------------------------------------------------------

def predict(model, X) -> np.ndarray:
    """Return 1-D numpy array of predictions."""
    pred = model.predict(X)
    return pred.ravel()


# ---------------------------------------------------------------------------
# Train all models in one call
# ---------------------------------------------------------------------------

def train_all(X_train, y_train, X_val, y_val, verbose: bool = True):
    """
    Train all four models and return a dict:
        {name: model}
    """
    results = {}

    if verbose:
        print("Training Linear Regression …")
    results["Linear Regression"] = train_linear_regression(X_train, y_train)

    if verbose:
        print("Training Random Forest …")
    results["Random Forest"] = train_random_forest(X_train, y_train)

    if verbose:
        print("Training Gradient Boosting …")
    results["Gradient Boosting"] = train_gradient_boosting(X_train, y_train)

    if verbose:
        print("Training Neural Network …")
    nn_model, backend = train_neural_network(X_train, y_train, X_val, y_val)
    results["Neural Network"] = nn_model
    if verbose:
        print(f"  → Neural Network backend: {backend}")

    return results
