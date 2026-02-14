"""Train a LightGBM fraud classifier on synthetic data and produce evaluation artifacts.

Reads:  fraud/demo/results/synthetic_data.csv
Writes: fraud/demo/results/summary.json
        fraud/demo/results/calibration_curve.png
"""

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add repo root so ds_tools is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ds_tools" / "src"))
from ds_tools.evaluation.calibration import plot_calibration
from ds_tools.evaluation.report import ClassificationEvaluator

RESULTS_DIR = Path(__file__).parent / "results"
FEATURES = [
    "transaction_amount",
    "hour_of_day",
    "day_of_week",
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
    "merchant_freq",
    "is_international",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RESULTS_DIR / "synthetic_data.csv")
    return df[FEATURES], df["is_fraud"]


def train_and_evaluate():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]

    # Evaluation via ds_tools
    evaluator = ClassificationEvaluator(y_test, y_prob, model_name="LightGBM (synthetic)")
    metrics = evaluator.summary()

    # Calculate medians for serving imputation
    numeric_cols = list(x_train.select_dtypes(include=[np.number]).columns)
    train_medians = x_train[numeric_cols].median().to_dict()

    # Save artifact for serving
    artifact = {
        "model": model,
        "feature_cols": FEATURES,
        "numeric_cols": numeric_cols,
        "train_medians": train_medians,
        "threshold": 0.5,
        "model_name": "LightGBM (synthetic)",
    }
    joblib.dump(artifact, RESULTS_DIR / "fraud_model.joblib")

    # Calibration plot
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_calibration(y_test.values, y_prob, model_name="LightGBM", ax=ax)
    fig.savefig(RESULTS_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary JSON
    summary = {
        "model": "LightGBM",
        "dataset": "synthetic (10,000 transactions, 5% fraud)",
        "train_size": len(x_train),
        "test_size": len(x_test),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "artifacts": ["calibration_curve.png", "summary.json"],
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  summary.json")
    print("  calibration_curve.png")
    print("  fraud_model.joblib (for serving details)")
    return summary


if __name__ == "__main__":
    train_and_evaluate()
