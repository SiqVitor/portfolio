"""Unit tests for ds_tools key functions.

Tests cover:
- FrequencyEncoder (fit, transform, unseen categories)
- expected_calibration_error
- brier_score
- ClassificationEvaluator summary output
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure ds_tools is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ds_tools" / "src"))

from ds_tools.evaluation.calibration import brier_score, expected_calibration_error
from ds_tools.evaluation.report import ClassificationEvaluator
from ds_tools.preprocessing.transformers import FrequencyEncoder


def test_frequency_encoder_fit_transform():
    """FrequencyEncoder should replace categories with their proportions."""
    df = pd.DataFrame({"color": ["red", "red", "blue", "green", "red"]})
    enc = FrequencyEncoder(columns=["color"], normalize=True)
    enc.fit(df)
    result = enc.transform(df)

    assert result["color"].iloc[0] == 0.6  # red: 3/5
    assert result["color"].iloc[2] == 0.2  # blue: 1/5
    assert result["color"].iloc[3] == 0.2  # green: 1/5


def test_frequency_encoder_unseen_category():
    """Unseen categories at transform time should default to 0.0."""
    df_train = pd.DataFrame({"color": ["red", "blue"]})
    df_test = pd.DataFrame({"color": ["red", "purple"]})

    enc = FrequencyEncoder(columns=["color"], normalize=True)
    enc.fit(df_train)
    result = enc.transform(df_test)

    assert result["color"].iloc[1] == 0.0  # purple not in training


def test_brier_score_perfect():
    """Perfect predictions should have Brier Score = 0."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_prob) == 0.0


def test_brier_score_worst():
    """Completely wrong predictions should have Brier Score = 1."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == 1.0


def test_ece_perfect_calibration():
    """Perfectly calibrated predictions should have ECE close to 0."""
    rng = np.random.RandomState(42)
    y_prob = rng.uniform(0, 1, size=1000)
    y_true = (rng.uniform(0, 1, size=1000) < y_prob).astype(int)

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05  # should be very small with enough samples


def test_evaluator_summary_keys():
    """ClassificationEvaluator.summary() should return the expected metric keys."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2])

    evaluator = ClassificationEvaluator(y_true, y_prob, model_name="Test")
    metrics = evaluator.summary()

    expected_keys = {"ROC-AUC", "Average Precision", "Log Loss", "Brier Score", "ECE"}
    assert set(metrics.keys()) == expected_keys
    assert all(isinstance(v, float) for v in metrics.values())
