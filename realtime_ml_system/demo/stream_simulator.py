"""Simulate a transaction event stream for online inference.

Generates synthetic transaction events one at a time, mimicking a Kafka
consumer reading from a topic. Each event is a dict with real-time features
extracted from the event payload.

In production, this would be a Kafka consumer. The demo uses an in-process
generator to avoid external dependencies.
"""

import numpy as np


def generate_feature_store(n_entities: int = 500, seed: int = 42) -> dict[str, dict]:
    """Pre-compute batch features, simulating a feature store (Redis/Feast).

    Returns a dict mapping entity_id → pre-computed feature dict.
    In production, these are refreshed every 6h by a batch pipeline.
    """
    rng = np.random.RandomState(seed)
    store = {}
    for entity_id in range(n_entities):
        store[entity_id] = {
            "avg_daily_spend_30d": round(rng.exponential(200), 2),
            "txn_count_7d": int(rng.poisson(15)),
            "distinct_merchants_30d": int(rng.poisson(8)),
            "merchant_freq": round(rng.uniform(0.001, 0.05), 6),
            "velocity_1h": int(rng.poisson(2)),
        }
    return store


def stream_events(n_events: int = 1000, fraud_rate: float = 0.05, seed: int = 42):
    """Yield transaction events one at a time, simulating a stream.

    Each event contains real-time features (from the event payload) and
    an entity_id for feature store lookup.
    """
    rng = np.random.RandomState(seed)

    for i in range(n_events):
        is_fraud = rng.random() < fraud_rate
        entity_id = rng.randint(0, 500)

        # Real-time features (extracted from event payload)
        amount = rng.lognormal(6.5, 0.8) if is_fraud else rng.lognormal(4.5, 1.0)
        event = {
            "event_id": i,
            "entity_id": entity_id,
            "transaction_amount": round(amount, 2),
            "hour_of_day": int(rng.choice(24)),
            "device_type": rng.choice(["mobile", "desktop", "tablet"]),
            "is_international": int(rng.random() < (0.3 if is_fraud else 0.05)),
            # Ground truth (not available at inference time — used for evaluation only)
            "_label": int(is_fraud),
        }
        yield event


if __name__ == "__main__":
    # Quick test: print first 5 events
    store = generate_feature_store()
    for i, event in enumerate(stream_events(n_events=5)):
        features = store.get(event["entity_id"], {})
        print(
            f"Event {event['event_id']}: amount=${event['transaction_amount']:.2f}, "
            f"store_features={len(features)} keys, label={event['_label']}"
        )
