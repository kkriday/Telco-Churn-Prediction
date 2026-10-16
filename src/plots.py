"""Report figures. One consistent style: white surface, quiet axes, fixed hue order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .config import FIG_DIR

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
NEUTRAL = "#a8a7a0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False, "axes.axisbelow": True, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def segment_churn(df, segments, overall_rate):
    """Churn rate by segment, small multiples sharing one x scale."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, (title, col, order) in zip(axes.ravel(), segments):
        rates = df.groupby(col)["Churn"].mean().reindex(order)
        counts = df[col].value_counts().reindex(order)
        y = np.arange(len(rates))[::-1]
        ax.barh(y, rates.values, color=SERIES[0], height=0.6)
        ax.axvline(overall_rate, color=INK_2, ls="--", lw=1)
        for yi, r, n in zip(y, rates.values, counts.values):
            ax.text(r + 0.01, yi, f"{r:.0%}  (n={n:,})", va="center", color=INK, fontsize=9)
        ax.set_yticks(y, rates.index)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 0.75)
    for ax in axes[1]:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Churn rate")
    axes[0, 0].text(overall_rate + 0.005, axes[0, 0].get_ylim()[1], f"overall {overall_rate:.0%}",
                    color=INK_2, fontsize=8, va="bottom")
    fig.suptitle("Who churns: churn rate by customer segment", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "segment_churn_rates.png")


def cv_comparison(cv_df):
    """Mean +/- std across CV folds, so overlapping models are visibly tied."""
    cv_df = cv_df[~cv_df["model"].str.startswith("Baseline")].sort_values("cv_pr_auc_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, metric, label in [(axes[0], "cv_roc_auc", "ROC-AUC"), (axes[1], "cv_pr_auc", "PR-AUC")]:
        y = np.arange(len(cv_df))
        ax.errorbar(cv_df[f"{metric}_mean"], y, xerr=cv_df[f"{metric}_std"], fmt="o",
                    color=SERIES[0], ecolor=SERIES[0], elinewidth=2, capsize=4, ms=8)
        for yi, m in zip(y, cv_df[f"{metric}_mean"]):
            ax.text(m, yi + 0.22, f"{m:.3f}", ha="center", fontsize=9, color=INK)
        ax.set_yticks(y, cv_df["model"])
        lo = (cv_df[f"{metric}_mean"] - cv_df[f"{metric}_std"]).min()
        hi = (cv_df[f"{metric}_mean"] + cv_df[f"{metric}_std"]).max()
        ax.set_xlim(lo - 0.01, hi + 0.01)
        ax.set_ylim(-0.6, len(cv_df) - 0.4)
        ax.set_title(f"{label} (5-fold CV, mean ± std)", loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "cv_model_comparison.png")


def roc_pr_curves(y_test, probas, prevalence):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[0].plot(fpr, tpr, lw=2, color=SERIES[i], label=name)
        axes[1].plot(rec, prec, lw=2, color=SERIES[i], label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1, label="Random")
    axes[1].axhline(prevalence, ls="--", color=NEUTRAL, lw=1, label="Random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    axes[0].set_title("ROC curve (held-out test set)", loc="left")
    axes[1].set_title("Precision-recall curve (held-out test set)", loc="left")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "roc_pr_curves.png")


def value_curve(thresholds, values, chosen, default=0.5):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(thresholds, values / 1000, color=SERIES[0], lw=2)
    ax.axhline(0, color=INK_2, lw=1)
    for t, label, color in [(chosen, "chosen", SERIES[1]), (default, "default 0.5", NEUTRAL)]:
        v = values[np.argmin(np.abs(thresholds - t))] / 1000
        ax.scatter([t], [v], s=60, color=color, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{label}: t={t:.2f}\n${v:,.1f}k", (t, v), textcoords="offset points",
                    xytext=(-20 if label == "chosen" else 6, -34), fontsize=9, color=INK,
                    ha="right" if label == "chosen" else "left")
    ax.set(xlabel="Decision threshold (predicted churn probability)",
           ylabel="Net campaign value ($k)")
    ax.set_title("Retention campaign value vs. threshold (test set)", loc="left", pad=12)
    ax.set_ylim(min(0, values.min() / 1000) - 2, values.max() / 1000 * 1.12)
    fig.tight_layout()
    _save(fig, "campaign_value_curve.png")


def gains_curve(y_test, proba):
    y = np.asarray(y_test)
    order = np.argsort(-proba)
    captured = np.cumsum(y[order]) / y.sum()
    share = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(share, captured, color=SERIES[0], lw=2, label="Model")
    ax.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1, label="Random targeting")
    for s in (0.2, 0.4):
        c = captured[int(s * len(y)) - 1]
        ax.scatter([s], [c], color=SERIES[0], s=50, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"top {s:.0%} -> {c:.0%} of churners", (s, c), textcoords="offset points",
                    xytext=(8, -14), fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(xlabel="Customers contacted (ranked by risk)", ylabel="Churners captured")
    ax.set_title("Cumulative gains (test set)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "cumulative_gains.png")


def confusion(y_test, proba, threshold):
    cm = confusion_matrix(y_test, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{labels[i][j]}", ha="center", va="center",
                    color=SURFACE if cm[i, j] > cm.max() / 2 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Stayed", "Churned"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.grid(False)
    ax.set_title(f"Confusion matrix at t={threshold:.2f}", loc="left")
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")
