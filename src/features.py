"""Feature engineering and the leak-free preprocessing step.

Everything that learns from data (scaling, encoding) lives inside a sklearn
ColumnTransformer, so it is fit on training folds only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ADD_ONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "num_add_ons", "avg_monthly_spend", "price_change",
]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only (no statistics learned across rows)."""
    X = X.copy()
    X["num_add_ons"] = (X[ADD_ONS] == "Yes").sum(axis=1)
    safe_tenure = X["tenure"].replace(0, np.nan)
    X["avg_monthly_spend"] = (X["TotalCharges"] / safe_tenure).fillna(X["MonthlyCharges"])
    # Positive when today's bill is higher than the customer's historical average.
    X["price_change"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
    X["tenure_band"] = pd.cut(
        X["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)
    X["auto_pay"] = X["PaymentMethod"].str.contains("automatic").map({True: "Yes", False: "No"})
    return X


def categorical_columns(X: pd.DataFrame):
    return [c for c in X.columns if c not in NUMERIC]


def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    cats = categorical_columns(add_features(X_sample))
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), cats),
        ],
        verbose_feature_names_out=False,
    )


def feature_step():
    return FunctionTransformer(add_features, validate=False)
