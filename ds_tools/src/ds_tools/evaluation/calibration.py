"""Probability-calibration metrics and reliability curves.

Calibration answers: *when the model says 70 %, is it really correct 70 % of the
time?*  Standard accuracy/AUC ignore this — a model can have high AUC but
produce wildly overconfident probabilities.

Metrics
-------
- **Brier Score**: Mean squared error between predicted probability and true
  label.  Lower is better; 0 = perfect.
- **Expected Calibration Error (ECE)**: Weighted average gap between predicted
  confidence and actual accuracy across probability bins.  Lower is better.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score — mean squared error of predicted probabilities.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : array-like of floats in [0, 1]

    Returns
    -------
    float  (lower is better, 0 = perfect)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Splits predictions into *n_bins* uniform-width bins and computes the
    weighted average of |accuracy − confidence| per bin.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : array-like of floats in [0, 1]
    n_bins : int

    Returns
    -------
    float  (lower is better, 0 = perfectly calibrated)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    return float(ece / len(y_true))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a reliability (calibration) curve with Brier and ECE annotations.

    Parameters
    ----------
    y_true, y_prob : array-like
    model_name : str   — label shown in legend
    n_bins : int       — number of equal-width bins
    ax : matplotlib Axes (optional)

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    bs = brier_score(y_true, y_prob)
    ece_val = expected_calibration_error(y_true, y_prob, n_bins)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(
        mean_predicted,
        fraction_pos,
        "s-",
        label=f"{model_name}\nBrier = {bs:.4f}  ECE = {ece_val:.4f}",
    )
    ax.fill_between(
        mean_predicted,
        fraction_pos,
        mean_predicted,
        alpha=0.15,
        color="tab:orange",
        label="Calibration gap",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (Reliability) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax
