"""Reusable ML visualisation helpers.

Functions are designed for binary classification workflows and accept raw
numpy arrays so they work with any framework (sklearn, LightGBM, XGBoost, …).

SHAP helpers assume a tree-based model compatible with ``shap.TreeExplainer``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def plot_shap_summary(model, X, feature_names=None, max_display: int = 20):
    """SHAP beeswarm summary plot for a tree-based model.

    Parameters
    ----------
    model : fitted tree model (LightGBM, XGBoost, sklearn GBT, …)
    X : array-like — samples to explain
    feature_names : list[str] or None
    max_display : int
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classifiers that return a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    return plt.gcf()


def plot_shap_waterfall(model, X, idx: int = 0):
    """SHAP waterfall plot for a single observation.

    Parameters
    ----------
    model : fitted tree model
    X : array-like — dataset (the *idx*-th row is explained)
    idx : int — row index to explain
    """
    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # binary classifier → take positive class
    if len(explanation.shape) == 3:
        explanation = explanation[:, :, 1]

    shap.plots.waterfall(explanation[idx], show=False)
    plt.tight_layout()
    return plt.gcf()


# ---------------------------------------------------------------------------
# ROC / PR overlays
# ---------------------------------------------------------------------------


def plot_roc_pr(y_true, y_prob_dict: dict, figsize=(14, 5)) -> plt.Figure:
    """Side-by-side ROC and Precision-Recall curves for multiple models.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob_dict : dict[str, array-like]
        Mapping ``model_name → predicted_probabilities``.
    """
    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.4f})")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax2.plot(rec, prec, linewidth=2, label=f"{name} (AP={ap:.4f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(
        title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances,
    feature_names,
    top_n: int = 20,
    figsize=(10, 8),
) -> plt.Figure:
    """Horizontal bar chart of feature importances (sorted)."""
    importances = np.asarray(importances)
    feature_names = np.asarray(feature_names)
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(idx)), importances[idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(feature_names[idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def plot_threshold_analysis(
    y_true,
    y_prob,
    figsize=(10, 6),
) -> tuple[plt.Figure, float]:
    """Precision, Recall, and F1 across classification thresholds.

    Returns
    -------
    (figure, best_f1_threshold)
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.arange(0.01, 1.0, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            precisions.append(np.nan)
            recalls.append(np.nan)
            f1s.append(np.nan)
        else:
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precisions, label="Precision", linewidth=1.5)
    ax.plot(thresholds, recalls, label="Recall", linewidth=1.5)
    ax.plot(thresholds, f1s, label="F1 Score", linewidth=2.5)

    best_t = thresholds[np.nanargmax(f1s)]
    ax.axvline(
        best_t,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 threshold = {best_t:.2f}",
    )

    ax.set(title="Threshold Analysis", xlabel="Threshold", ylabel="Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, float(best_t)
