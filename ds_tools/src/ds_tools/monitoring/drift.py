"""Data-drift detection, simulation, and reporting.

In production ML systems, model performance degrades when the input
distribution shifts away from what the model saw during training.  This
module provides:

- **PSI** (Population Stability Index) — fast, binned divergence metric.
- **KS test** — non-parametric two-sample test.
- **simulate_drift** — artificially inject distributional changes into a
  DataFrame so you can *demonstrate* monitoring without a live system.
- **drift_report** — multi-feature summary table with alerts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Population Stability Index between two distributions.

    Interpretation (industry rule of thumb):
    - PSI < 0.10  → no significant shift
    - 0.10 ≤ PSI < 0.20 → moderate shift — investigate
    - PSI ≥ 0.20 → significant shift — retrain

    Parameters
    ----------
    reference, current : array-like of floats
    n_bins : int — number of quantile-based bins (from reference)
    eps : float — smoothing constant to avoid log(0)
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Bins from reference distribution
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    ref_counts = np.histogram(reference, bins=bin_edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=bin_edges)[0] / len(current)

    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Kolmogorov-Smirnov two-sample test for distributional drift.

    Returns
    -------
    dict with 'statistic', 'p_value', 'is_drift'
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_drift": p_value < threshold,
    }


# ---------------------------------------------------------------------------
# Drift simulation
# ---------------------------------------------------------------------------


def simulate_drift(
    df: pd.DataFrame,
    feature: str,
    drift_type: str = "shift",
    magnitude: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Artificially inject data drift into a DataFrame.

    This lets you demonstrate monitoring pipelines without a live production
    system.

    Parameters
    ----------
    df : DataFrame
    feature : str — column to perturb
    drift_type : {'shift', 'scale', 'spike', 'missing'}
        - ``shift`` — add ``magnitude × std`` to every value (covariate drift)
        - ``scale`` — multiply by ``1 + magnitude`` (variance change)
        - ``spike`` — inject extreme values in ~10 %·magnitude of rows
        - ``missing`` — set ~10 %·magnitude of rows to NaN
    magnitude : float — strength of the perturbation (1.0 = moderate)
    seed : int
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    if drift_type == "shift":
        sigma = df[feature].std()
        df[feature] = df[feature] + magnitude * sigma

    elif drift_type == "scale":
        df[feature] = df[feature] * (1 + magnitude)

    elif drift_type == "spike":
        n_spikes = int(len(df) * min(magnitude * 0.1, 0.5))
        spike_idx = rng.choice(df.index, size=n_spikes, replace=False)
        spike_val = df[feature].quantile(0.99) * (1 + magnitude)
        df.loc[spike_idx, feature] = spike_val

    elif drift_type == "missing":
        n_miss = int(len(df) * min(magnitude * 0.1, 0.5))
        miss_idx = rng.choice(df.index, size=n_miss, replace=False)
        df.loc[miss_idx, feature] = np.nan

    else:
        raise ValueError(f"Unknown drift_type: {drift_type!r}")

    return df


# ---------------------------------------------------------------------------
# Multi-feature drift report
# ---------------------------------------------------------------------------


def drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate a drift report for multiple features.

    Returns a DataFrame sorted by PSI (descending) with columns:
    feature, psi, psi_alert, ks_statistic, ks_p_value, ks_drift,
    ref_mean, cur_mean, mean_shift_pct.
    """
    rows = []
    for feat in features:
        ref = reference_df[feat].dropna().values
        cur = current_df[feat].dropna().values

        psi_val = psi(ref, cur, n_bins=n_bins)
        ks = ks_drift_test(ref, cur)

        rows.append(
            {
                "feature": feat,
                "psi": round(psi_val, 4),
                "psi_alert": (
                    "HIGH" if psi_val >= 0.2 else "MEDIUM" if psi_val >= 0.1 else "LOW"
                ),
                "ks_statistic": round(ks["statistic"], 4),
                "ks_p_value": ks["p_value"],
                "ks_drift": ks["is_drift"],
                "ref_mean": round(float(ref.mean()), 4),
                "cur_mean": round(float(cur.mean()), 4),
                "mean_shift_pct": round(
                    (cur.mean() - ref.mean()) / (abs(ref.mean()) + 1e-10) * 100, 2
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
