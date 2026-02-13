"""Evaluation metrics — calibration, classification reports, hard-sample analysis."""

from .calibration import brier_score, expected_calibration_error, plot_calibration
from .report import ClassificationEvaluator

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "plot_calibration",
    "ClassificationEvaluator",
]
