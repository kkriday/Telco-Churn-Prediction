"""End-to-end churn pipeline: EDA figures, tuned models, threshold choice, SHAP.

Usage:
    python run_pipeline.py

Outputs land in reports/ (metrics.json, model_comparison.csv, figures/).
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)

from src import plots
from src.config import (CV_FOLDS, MONTHS_RETAINED, OFFER_COST, REPORTS_DIR,
                        ROOT, SAVE_RATE, SEARCH_ITERATIONS, SEED, TEST_SIZE)
from src.data import load_clean, split_xy
from src.evaluate import (best_threshold, campaign_value, classification_metrics,
                          gains, value_curve)
from src.features import add_features
from src.models import candidates, make_pipeline

warnings.filterwarnings("ignore", category=UserWarning)
BASELINE = "Baseline (majority class)"
SIMPLICITY_ORDER = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]


def eda(df):
    seg = add_features(df)
    seg["Churn"] = df["Churn"]
    overall = df["Churn"].mean()
    plots.segment_churn(seg, [
        ("Contract type", "Contract", ["Month-to-month", "One year", "Two year"]),
        ("Tenure", "tenure_band", ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]),
        ("Internet service", "InternetService", ["Fiber optic", "DSL", "No"]),
        ("Payment method", "PaymentMethod", ["Electronic check", "Mailed check",
                                             "Bank transfer (automatic)", "Credit card (automatic)"]),
    ], overall)
    return {
        "rows": int(len(df)),
        "churn_rate": float(overall),
        "churn_by_contract": seg.groupby("Contract")["Churn"].mean().round(4).to_dict(),
        "churn_by_tenure_band": seg.groupby("tenure_band")["Churn"].mean().round(4).to_dict(),
        "churn_by_internet": seg.groupby("InternetService")["Churn"].mean().round(4).to_dict(),
        "churn_by_payment": seg.groupby("PaymentMethod")["Churn"].mean().round(4).to_dict(),
    }


def tune_all(X_train, y_train):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    fitted, rows = {}, []
    for name, (est, space) in candidates(pos_weight).items():
        print(f"Tuning {name} ...", flush=True)
        search = RandomizedSearchCV(
            make_pipeline(X_train, est), space or {"model__strategy": ["prior"]},
            n_iter=SEARCH_ITERATIONS if space else 1, scoring=scoring, refit="pr_auc",
            cv=cv, random_state=SEED, n_jobs=1,
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        r = search.cv_results_
        rows.append({
            "model": name,
            "cv_roc_auc_mean": r["mean_test_roc_auc"][i], "cv_roc_auc_std": r["std_test_roc_auc"][i],
            "cv_pr_auc_mean": r["mean_test_pr_auc"][i], "cv_pr_auc_std": r["std_test_pr_auc"][i],
            "best_params": {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                            for k, v in search.best_params_.items()},
        })
        fitted[name] = search.best_estimator_
    return fitted, pd.DataFrame(rows), cv


def explain(model, X_test, name):
    """SHAP beeswarm for the selected model."""
    import matplotlib.pyplot as plt

    prep = model[:-1]
    Xt = pd.DataFrame(prep.transform(X_test), columns=prep[-1].get_feature_names_out())
    est = model[-1]
    if name == "Logistic Regression":
        explainer = shap.LinearExplainer(est, Xt)
    else:
        explainer = shap.TreeExplainer(est)
    sv = explainer(Xt)
    if sv.values.ndim == 3:  # some tree models return one slice per class
        sv = sv[:, :, 1]
    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    fig.axes[0].set_title(f"What drives predicted churn ({name}, SHAP)", loc="left",
                          fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots.FIG_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=Xt.columns).sort_values(ascending=False)
    return mean_abs.head(10).round(4).to_dict()


def main():
    df = load_clean()
    summary = {"data": eda(df)}

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    fitted, cv_df, cv = tune_all(X_train, y_train)
    plots.cv_comparison(cv_df)

    # One-standard-error rule: among models whose CV PR-AUC is within one std of
    # the top score, take the simplest. Ties go to the model that is cheapest to
    # run and easiest to explain to a retention team.
    contenders = cv_df[cv_df.model != BASELINE].set_index("model")
    top = contenders["cv_pr_auc_mean"].idxmax()
    bar = contenders.loc[top, "cv_pr_auc_mean"] - contenders.loc[top, "cv_pr_auc_std"]
    eligible = contenders[contenders["cv_pr_auc_mean"] >= bar].index
    best_name = next(m for m in SIMPLICITY_ORDER if m in eligible)
    best = fitted[best_name]
    print(f"Top CV score: {top}; selected (1-SE rule): {best_name}")

    # Threshold is chosen on out-of-fold training predictions, never on the test set.
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof, X_train["MonthlyCharges"])

    probas = {n: m.predict_proba(X_test)[:, 1] for n, m in fitted.items()}
    test_rows = []
    for n, p in probas.items():
        m = classification_metrics(y_test, p, 0.5)
        test_rows.append({"model": n, **{f"test_{k}": v for k, v in m.items()}})
    results = cv_df.merge(pd.DataFrame(test_rows), on="model")
    results.drop(columns="best_params").round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    p_best = probas[best_name]
    charges = X_test["MonthlyCharges"]
    strategies = {
        "No campaign": None,
        "Offer everyone": 0.0,
        "Model, default threshold 0.5": 0.5,
        f"Model, value-optimized threshold {threshold:.2f}": threshold,
    }
    value_table = {}
    for label, t in strategies.items():
        if t is None:
            value_table[label] = {"net_value": 0.0, "customers_contacted": 0}
        else:
            value_table[label] = {
                "net_value": round(campaign_value(y_test, p_best, charges, t), 0),
                "customers_contacted": int((p_best >= t).sum()),
            }

    t_grid, v_grid = value_curve(y_test, p_best, charges)
    plots.value_curve(t_grid, v_grid, threshold)
    plots.gains_curve(y_test, p_best)
    plots.confusion(y_test, p_best, threshold)
    plots.roc_pr_curves(y_test, {n: p for n, p in probas.items() if n != BASELINE}, y_test.mean())

    summary.update({
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        "selected_model": best_name,
        "top_cv_model": top,
        "selected_params": cv_df.set_index("model").loc[best_name, "best_params"],
        "threshold": threshold,
        "test_at_threshold": {k: round(v, 4) for k, v in
                              classification_metrics(y_test, p_best, threshold).items()},
        "test_at_0.5": {k: round(v, 4) for k, v in
                        classification_metrics(y_test, p_best, 0.5).items()},
        "gains": {"top_10pct": round(gains(y_test, p_best, 0.1), 4),
                  "top_20pct": round(gains(y_test, p_best, 0.2), 4),
                  "top_40pct": round(gains(y_test, p_best, 0.4), 4)},
        "campaign": {"assumptions": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE,
                                     "months_retained": MONTHS_RETAINED},
                     "strategies": value_table},
        "top_shap_features": explain(best, X_test, best_name),
        "cv_results": cv_df.round(4).to_dict(orient="records"),
    })
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "churn_model.joblib")
    print(json.dumps({k: summary[k] for k in
                      ["selected_model", "threshold", "test_at_threshold", "test_at_0.5",
                       "gains", "campaign", "top_shap_features"]}, indent=2, default=str))
    print(results.drop(columns="best_params").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
