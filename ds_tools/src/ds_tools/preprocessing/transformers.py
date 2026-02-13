"""Sklearn-compatible preprocessing transformers.

All transformers follow the `fit` / `transform` API so they can be dropped
straight into a `Pipeline` or `ColumnTransformer`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace categorical values with their frequency in the training set.

    Using frequency (or count) encoding avoids the dimensionality explosion of
    one-hot encoding on high-cardinality features while preserving ordinal
    information about category popularity.

    Parameters
    ----------
    columns : list[str] or None
        Columns to encode.  ``None`` → all object/category columns.
    normalize : bool
        If True return proportions; otherwise raw counts.
    """

    def __init__(self, columns=None, normalize: bool = True):
        self.columns = columns
        self.normalize = normalize

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = (
            self.columns
            if self.columns is not None
            else X.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        self.freq_maps_: dict[str, dict] = {}
        for col in cols:
            freq = X[col].value_counts(normalize=self.normalize)
            self.freq_maps_[col] = freq.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, freq_map in self.freq_maps_.items():
            default = 0.0 if self.normalize else 1
            X[col] = X[col].map(freq_map).fillna(default)
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to bounds learned at fit time.

    Two strategies:
    - ``iqr``:  lower = Q1 − factor·IQR, upper = Q3 + factor·IQR
    - ``percentile``: lower = P(lower_pct), upper = P(upper_pct)

    Parameters
    ----------
    method : {'iqr', 'percentile'}
    factor : float  — IQR multiplier (only used when method='iqr')
    lower_pct, upper_pct : float  — percentile bounds (only method='percentile')
    """

    def __init__(
        self,
        method: str = "iqr",
        factor: float = 1.5,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
    ):
        self.method = method
        self.factor = factor
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.method == "iqr":
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - self.factor * iqr
            self.upper_ = q3 + self.factor * iqr
        else:
            self.lower_ = np.nanpercentile(X, self.lower_pct, axis=0)
            self.upper_ = np.nanpercentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)


class BalanceDeltaTransformer(BaseEstimator, TransformerMixin):
    """Derive balance-discrepancy features for transactional fraud detection.

    In legitimate transactions the post-transaction balance should equal
    ``old_balance − amount`` (for the sender).  Any deviation is a strong
    fraud signal — the PaySim dataset deliberately injects these anomalies.

    Generated features
    ------------------
    - ``orig_balance_delta``  — actual new balance minus expected
    - ``orig_balance_error``  — absolute value of the delta
    - ``orig_balance_zeroed`` — 1 if sender balance went to zero
    - ``dest_balance_delta``  — same for the receiver side
    - ``dest_balance_error``
    """

    def __init__(
        self,
        amount_col: str = "amount",
        old_orig_col: str = "oldbalanceOrg",
        new_orig_col: str = "newbalanceOrig",
        old_dest_col: str = "oldbalanceDest",
        new_dest_col: str = "newbalanceDest",
    ):
        self.amount_col = amount_col
        self.old_orig_col = old_orig_col
        self.new_orig_col = new_orig_col
        self.old_dest_col = old_dest_col
        self.new_dest_col = new_dest_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # Sender side
        expected_orig = X[self.old_orig_col] - X[self.amount_col]
        X["orig_balance_delta"] = X[self.new_orig_col] - expected_orig
        X["orig_balance_error"] = X["orig_balance_delta"].abs()
        X["orig_balance_zeroed"] = (X[self.new_orig_col] == 0).astype(int)
        # Receiver side
        expected_dest = X[self.old_dest_col] + X[self.amount_col]
        X["dest_balance_delta"] = X[self.new_dest_col] - expected_dest
        X["dest_balance_error"] = X["dest_balance_delta"].abs()
        return X
