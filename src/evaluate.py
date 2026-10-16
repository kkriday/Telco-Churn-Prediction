"""Metrics, business value, and threshold selection."""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .config import MONTHS_RETAINED, OFFER_COST, SAVE_RATE


def classification_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "flagged_share": pred.mean(),
    }


def campaign_value(y_true, proba, monthly_charges, threshold):
    """Net dollars from offering retention to every customer above threshold.

    Saved churner:  SAVE_RATE * MONTHS_RETAINED * MonthlyCharges - OFFER_COST
    Loyal customer: -OFFER_COST (offer spent on someone who would have stayed)
    Unflagged:      0 (same as running no campaign)
    """
    y = np.asarray(y_true)
    m = np.asarray(monthly_charges)
    flagged = np.asarray(proba) >= threshold
    gain = SAVE_RATE * MONTHS_RETAINED * m - OFFER_COST
    return float(np.where(flagged & (y == 1), gain, 0).sum()
                 - (flagged & (y == 0)).sum() * OFFER_COST)


def value_curve(y_true, proba, monthly_charges, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    values = np.array([campaign_value(y_true, proba, monthly_charges, t) for t in thresholds])
    return thresholds, values


def best_threshold(y_true, proba, monthly_charges):
    t, v = value_curve(y_true, proba, monthly_charges)
    return float(t[int(np.argmax(v))])


def gains(y_true, proba, share):
    """Share of all churners captured in the top `share` of risk scores."""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba))
    k = int(round(share * len(y)))
    return y[order[:k]].sum() / y.sum()
