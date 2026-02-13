"""ClassificationEvaluator — one-call full evaluation report.

Usage
-----
>>> evaluator = ClassificationEvaluator(y_true, y_prob, model_name="LightGBM")
>>> evaluator.summary()          # prints scalar metrics + classification report
>>> evaluator.plot_full_report()  # 4-panel figure (ROC, PR, CM, Calibration)
>>> evaluator.hard_samples(X)     # worst misclassifications with individual log-loss
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
)

from .calibration import brier_score, expected_calibration_error, plot_calibration


class ClassificationEvaluator:
    """End-to-end evaluation for binary classifiers.

    Computes scalar metrics, generates publication-quality figures, and
    identifies the hardest misclassified samples (highest individual
    cross-entropy loss).

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : array-like of floats in [0, 1] — predicted P(positive)
    threshold : float — decision boundary (default 0.5)
    model_name : str  — label used in titles and legends
    """

    def __init__(
        self,
        y_true,
        y_prob,
        threshold: float = 0.5,
        model_name: str = "Model",
    ):
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.threshold = threshold
        self.y_pred = (self.y_prob >= threshold).astype(int)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Scalar summary
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Print and return key evaluation metrics."""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(self.y_true, self.y_prob)
        bs = brier_score(self.y_true, self.y_prob)
        ece_val = expected_calibration_error(self.y_true, self.y_prob)
        ll = log_loss(self.y_true, self.y_prob)

        metrics = {
            "ROC-AUC": roc_auc,
            "Average Precision": ap,
            "Log Loss": ll,
            "Brier Score": bs,
            "ECE": ece_val,
        }

        print(f"\n{'=' * 55}")
        print(f"  {self.model_name} — Evaluation Report  (threshold={self.threshold})")
        print(f"{'=' * 55}")
        for k, v in metrics.items():
            print(f"  {k:22s} {v:.6f}")
        print(f"{'=' * 55}\n")
        print(
            classification_report(
                self.y_true,
                self.y_pred,
                target_names=["Legitimate", "Fraud"],
            )
        )
        return metrics

    # ------------------------------------------------------------------
    # 4-panel figure
    # ------------------------------------------------------------------
    def plot_full_report(self, figsize=(16, 12)) -> plt.Figure:
        """ROC, PR, Confusion Matrix, Calibration in a single figure."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # --- ROC ---
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0, 0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # --- PR ---
        prec, rec, _ = precision_recall_curve(self.y_true, self.y_prob)
        ap = average_precision_score(self.y_true, self.y_prob)
        axes[0, 1].plot(rec, prec, linewidth=2, label=f"AP = {ap:.4f}")
        axes[0, 1].set(
            title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision"
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # --- Confusion Matrix ---
        cm = confusion_matrix(self.y_true, self.y_pred)
        axes[1, 0].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[1, 0].set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                axes[1, 0].text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=color,
                )
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(["Legit", "Fraud"])
        axes[1, 0].set_yticklabels(["Legit", "Fraud"])
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Actual")

        # --- Calibration ---
        plot_calibration(self.y_true, self.y_prob, self.model_name, ax=axes[1, 1])

        fig.suptitle(f"{self.model_name} — Full Evaluation", fontsize=14, y=1.01)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Hard-sample analysis
    # ------------------------------------------------------------------
    def hard_samples(self, X=None, n: int = 10) -> dict:
        """Identify the *n* misclassifications with highest individual log-loss.

        These are predictions where the model was *most confidently wrong*
        (e.g., predicted 0.01 for a real fraud).

        Parameters
        ----------
        X : DataFrame or array (optional) — if given, returns the corresponding rows.
        n : int — number of hard samples to return.

        Returns
        -------
        dict with keys 'indices', 'true_labels', 'pred_probs', 'individual_loss',
        and optionally 'features' (if X provided).
        """
        eps = 1e-15
        p = np.clip(self.y_prob, eps, 1 - eps)
        individual_loss = -(self.y_true * np.log(p) + (1 - self.y_true) * np.log(1 - p))

        misclassified = self.y_pred != self.y_true
        hard_idx = np.where(misclassified)[0]
        hard_losses = individual_loss[hard_idx]
        top_n = hard_idx[np.argsort(hard_losses)[::-1][:n]]

        results = {
            "indices": top_n,
            "true_labels": self.y_true[top_n],
            "pred_probs": self.y_prob[top_n],
            "individual_loss": individual_loss[top_n],
        }
        if X is not None:
            import pandas as pd

            if isinstance(X, pd.DataFrame):
                results["features"] = X.iloc[top_n]
            else:
                results["features"] = X[top_n]

        # Pretty-print
        print(f"\n{'=' * 65}")
        print(f"  Top {n} Hardest Misclassifications  ({self.model_name})")
        print(f"{'=' * 65}")
        for idx in top_n:
            label = "FP" if self.y_true[idx] == 0 else "FN"
            print(
                f"  [{label}]  idx={idx:>7d}  true={self.y_true[idx]}  "
                f"prob={self.y_prob[idx]:.4f}  loss={individual_loss[idx]:.4f}"
            )

        return results
