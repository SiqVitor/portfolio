"""Online inference pipeline — batch-train then stream-score.

1. Batch phase: generate synthetic data, train LightGBM, save model
2. Online phase: simulate event stream, assemble features, score, log latency

Output: realtime_ml_system/demo/results/summary.json
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Ensure ds_tools and local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ds_tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from stream_simulator import generate_feature_store, stream_events

RESULTS_DIR = Path(__file__).parent / "results"

# Same feature columns used in batch training and online scoring
FEATURE_COLS = [
    "transaction_amount",
    "hour_of_day",
    "is_international",
    "avg_daily_spend_30d",
    "txn_count_7d",
    "distinct_merchants_30d",
    "merchant_freq",
    "velocity_1h",
]


def batch_train(seed: int = 42) -> lgb.LGBMClassifier:
    """Train a model on synthetic historical data (batch phase)."""
    rng = np.random.RandomState(seed)
    n = 10_000
    fraud_rate = 0.05
    n_fraud = int(n * fraud_rate)
    labels = np.array([0] * (n - n_fraud) + [1] * n_fraud)
    rng.shuffle(labels)

    feature_store = generate_feature_store(seed=seed)

    rows = []
    for i in range(n):
        is_fraud = labels[i]
        entity_id = rng.randint(0, 500)
        store_feats = feature_store.get(entity_id, {})
        amount = rng.lognormal(6.5, 0.8) if is_fraud else rng.lognormal(4.5, 1.0)
        rows.append(
            {
                "transaction_amount": round(amount, 2),
                "hour_of_day": int(rng.choice(24)),
                "is_international": int(rng.random() < (0.3 if is_fraud else 0.05)),
                **store_feats,
            }
        )

    df = pd.DataFrame(rows)
    x = df[FEATURE_COLS]
    y = pd.Series(labels)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
    print(f"  Batch training complete — validation AUC: {val_auc:.4f}")
    return model


def assemble_features(event: dict, feature_store: dict) -> pd.DataFrame:
    """Merge real-time features with pre-computed features into a model-ready vector."""
    store_feats = feature_store.get(
        event["entity_id"],
        {
            "avg_daily_spend_30d": 0.0,
            "txn_count_7d": 0,
            "distinct_merchants_30d": 0,
            "merchant_freq": 0.0,
            "velocity_1h": 0,
        },
    )
    row = {
        "transaction_amount": event["transaction_amount"],
        "hour_of_day": event["hour_of_day"],
        "is_international": event["is_international"],
        **store_feats,
    }
    return pd.DataFrame([{c: row[c] for c in FEATURE_COLS}])


def online_inference(model: lgb.LGBMClassifier, n_events: int = 1000):
    """Simulate online scoring: stream events → assemble features → predict → log."""
    feature_store = generate_feature_store(seed=42)
    latencies = []
    predictions = []
    labels = []

    for event in stream_events(n_events=n_events, seed=99):
        t0 = time.perf_counter()

        features = assemble_features(event, feature_store)
        prob = float(model.predict_proba(features)[:, 1][0])

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        predictions.append(prob)
        labels.append(event["_label"])

    return latencies, predictions, labels


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: Batch Training ===")
    model = batch_train()

    print("\n=== Phase 2: Online Inference (streaming) ===")
    n_events = 1000
    latencies, predictions, labels = online_inference(model, n_events=n_events)

    # Compute metrics
    latencies_arr = np.array(latencies)
    predictions_arr = np.array(predictions)
    labels_arr = np.array(labels)

    threshold = 0.5
    fraud_detected = int((predictions_arr >= threshold).sum())
    auc = roc_auc_score(labels_arr, predictions_arr) if labels_arr.sum() > 0 else 0.0
    total_time_sec = latencies_arr.sum() / 1000

    summary = {
        "events_processed": n_events,
        "fraud_detected": fraud_detected,
        "actual_fraud": int(labels_arr.sum()),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 4),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 4),
        "latency_mean_ms": round(float(latencies_arr.mean()), 4),
        "throughput_events_per_sec": round(n_events / total_time_sec, 1),
        "model_roc_auc": round(auc, 6),
        "avg_prediction": round(float(predictions_arr.mean()), 6),
        "prediction_std": round(float(predictions_arr.std()), 6),
    }

    output_path = RESULTS_DIR / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Events processed:  {summary['events_processed']}")
    print(f"  Fraud detected:    {summary['fraud_detected']} (actual: {summary['actual_fraud']})")
    print(f"  Latency p50:       {summary['latency_p50_ms']:.2f} ms")
    print(f"  Latency p95:       {summary['latency_p95_ms']:.2f} ms")
    print(f"  Throughput:        {summary['throughput_events_per_sec']:.0f} events/sec")
    print(f"  Model AUC:         {summary['model_roc_auc']:.4f}")
    print(f"\n  Results saved to {output_path}")
    return summary


if __name__ == "__main__":
    run()
