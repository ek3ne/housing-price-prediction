"""
preprocessing.py
Handles missing values, categorical encoding, and feature scaling for the
Ames Housing dataset (train.csv / test.csv).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Columns where NA actually means "None" / "No feature present"
# (per the Ames Housing data description)
# ---------------------------------------------------------------------------
_NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
    "MasVnrType",
]

_ZERO_COLS = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
              "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
              "GarageYrBlt", "GarageCars", "GarageArea"]


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in _NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in _ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: fill per neighbourhood median
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # Remaining numerics → median; categoricals → mode
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSF"]       = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["HouseAge"]      = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"]      = df["YrSold"] - df["YearRemodAdd"]
    return df


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_val, y_train, y_val : numpy arrays (scaled)
    feature_names                   : list[str]
    scaler                          : fitted StandardScaler
    """
    from global_imports import df as raw_df

    data = raw_df.copy()

    # Drop Id – not a feature
    data = data.drop(columns=["Id"], errors="ignore")

    # Separate target before any transforms that could leak
    y = np.log1p(data.pop("SalePrice"))          # log-transform for better regression

    # Fill → engineer → encode
    data = _fill_missing(data)
    data = _engineer_features(data)
    data = _encode_categoricals(data)

    X = data.values.astype(np.float32)
    feature_names = data.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.values, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, feature_names, scaler


def prepare_test_data(scaler, train_feature_names: list):
    """
    Preprocess test.csv using the already-fitted scaler.
    Aligns columns to match training data.
    """
    from global_imports import df_test as raw_test

    data = raw_test.copy()
    ids  = data.pop("Id") if "Id" in data.columns else None

    data = _fill_missing(data)
    data = _engineer_features(data)
    data = _encode_categoricals(data)

    # Align columns: add missing cols as 0, drop extra cols
    for col in train_feature_names:
        if col not in data.columns:
            data[col] = 0
    data = data[train_feature_names]

    X_test = scaler.transform(data.values.astype(np.float32))
    return X_test, ids
