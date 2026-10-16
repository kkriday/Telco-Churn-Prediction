"""Candidate models and their hyperparameter search spaces."""
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import SEED
from .features import build_preprocessor, feature_step


def make_pipeline(X_sample, estimator) -> Pipeline:
    return Pipeline([
        ("features", feature_step()),
        ("prep", build_preprocessor(X_sample)),
        ("model", estimator),
    ])


def candidates(pos_weight: float):
    """name -> (estimator, search space). An empty space means no tuning.

    pos_weight is the negative-to-positive ratio of the training labels, used by
    XGBoost to offset class imbalance the way class_weight does elsewhere.
    """
    return {
        "Baseline (majority class)": (
            DummyClassifier(strategy="prior"), {},
        ),
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=5000),
            {"model__C": loguniform(1e-3, 10)},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED),
            {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": randint(4, 16),
                "model__min_samples_leaf": randint(1, 30),
                "model__max_features": uniform(0.2, 0.6),
            },
        ),
        "XGBoost": (
            XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                          n_jobs=-1, random_state=SEED),
            {
                "model__n_estimators": randint(100, 600),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(2, 7),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 20),
            },
        ),
        "LightGBM": (
            LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
            {
                "model__n_estimators": randint(100, 600),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__num_leaves": randint(8, 64),
                "model__min_child_samples": randint(10, 100),
                "model__subsample": uniform(0.6, 0.4),
                "model__subsample_freq": [1],
                "model__colsample_bytree": uniform(0.5, 0.5),
            },
        ),
    }
