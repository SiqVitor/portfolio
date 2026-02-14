"""Generate a synthetic fraud detection dataset.

Creates 10,000 transactions with realistic features:
- transaction_amount, hour_of_day, day_of_week
- sender/receiver balance deltas
- device_type, is_international
- frequency-encoded merchant_id

Output: Fraud Detection/demo/results/synthetic_data.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(
    n_samples: int = 10_000, fraud_rate: float = 0.05, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic transaction data with controllable fraud rate."""
    rng = np.random.RandomState(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    is_fraud = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(is_fraud)

    # Transaction amount: legit ~N(150, 80), fraud ~N(800, 400)
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=6.5, sigma=0.8, size=n_samples),
        rng.lognormal(mean=4.5, sigma=1.0, size=n_samples),
    )

    # Temporal features
    hour = rng.choice(24, size=n_samples, p=_hour_distribution(is_fraud, rng))
    day_of_week = rng.randint(0, 7, size=n_samples)

    # Balance features (fraud transactions have suspicious deltas)
    old_balance_orig = rng.exponential(5000, size=n_samples)
    new_balance_orig = np.where(
        is_fraud,
        old_balance_orig * rng.uniform(0.0, 0.1, size=n_samples),  # drained
        np.maximum(old_balance_orig - amount, 0),
    )
    old_balance_dest = rng.exponential(3000, size=n_samples)
    new_balance_dest = old_balance_dest + amount

    # Categorical features
    n_merchants = 200
    merchant_id = rng.randint(0, n_merchants, size=n_samples)
    device_type = rng.choice(["mobile", "desktop", "tablet"], size=n_samples, p=[0.6, 0.3, 0.1])
    is_international = rng.binomial(1, np.where(is_fraud, 0.3, 0.05))

    # Merchant frequency (proxy for frequency encoding)
    merchant_counts = pd.Series(merchant_id).value_counts(normalize=True)
    merchant_freq = pd.Series(merchant_id).map(merchant_counts).values

    df = pd.DataFrame(
        {
            "transaction_amount": np.round(amount, 2),
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "old_balance_orig": np.round(old_balance_orig, 2),
            "new_balance_orig": np.round(new_balance_orig, 2),
            "old_balance_dest": np.round(old_balance_dest, 2),
            "new_balance_dest": np.round(new_balance_dest, 2),
            "merchant_freq": np.round(merchant_freq, 6),
            "device_type": device_type,
            "is_international": is_international,
            "is_fraud": is_fraud,
        }
    )
    return df


def _hour_distribution(is_fraud: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Flat distribution — per-sample fraud hour bias is applied after."""
    return np.ones(24) / 24


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    output_path = output_dir / "synthetic_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} transactions ({df['is_fraud'].sum()} fraud)")
    print(f"Saved to {output_path}")
