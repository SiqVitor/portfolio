"""Fraud Detection — FastAPI inference endpoint.

Run:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000

Request:
    POST /predict
    {
        "amount": 181000.0,
        "oldbalanceOrg": 181000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "type": "TRANSFER",
        "step": 1
    }
"""

from __future__ import annotations

import os
import time
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Load artefacts ───────────────────────────────────────────────────
ARTEFACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "artefacts", "fraud_model.joblib"
)
artefacts = joblib.load(ARTEFACT_PATH)

MODEL = artefacts["model"]
FEATURE_COLS = artefacts["feature_cols"]
THRESHOLD = artefacts["threshold"]
MODEL_NAME = artefacts["model_name"]

logger.info(
    "Loaded %s  |  threshold=%.2f  |  %d features",
    MODEL_NAME,
    THRESHOLD,
    len(FEATURE_COLS),
)

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="Fraud Detection API", version="0.1.0")


class Transaction(BaseModel):
    """Input schema matching PaySim fields."""

    step: int = 1
    type: str = "TRANSFER"
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0


class Prediction(BaseModel):
    is_fraud: bool
    probability: float
    threshold: float
    model: str
    latency_ms: float


def _engineer(txn: Transaction) -> pd.DataFrame:
    """Replicate the feature engineering from the training notebook."""
    row = {
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": txn.newbalanceOrig,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
    }

    expected_orig = row["oldbalanceOrg"] - row["amount"]
    row["orig_balance_delta"] = row["newbalanceOrig"] - expected_orig
    row["orig_balance_error"] = abs(row["orig_balance_delta"])
    row["orig_balance_zeroed"] = int(row["newbalanceOrig"] == 0)

    expected_dest = row["oldbalanceDest"] + row["amount"]
    row["dest_balance_delta"] = row["newbalanceDest"] - expected_dest
    row["dest_balance_error"] = abs(row["dest_balance_delta"])

    row["log_amount"] = float(np.log1p(row["amount"]))
    row["amount_to_balance_ratio"] = row["amount"] / (row["oldbalanceOrg"] + 1)
    row["is_high_risk_type"] = int(txn.type in ("TRANSFER", "CASH_OUT"))

    # Type dummies (must match training)
    for t in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        row[f"type_{t}"] = int(txn.type == t)

    row["hour_of_day"] = txn.step % 24

    return pd.DataFrame([row])[FEATURE_COLS]


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction) -> Prediction:
    t0 = time.perf_counter()

    X = _engineer(txn)
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    is_fraud = prob >= THRESHOLD

    latency = (time.perf_counter() - t0) * 1000
    logger.info(
        "amount=%.2f  type=%s  prob=%.4f  fraud=%s  latency=%.1fms",
        txn.amount,
        txn.type,
        prob,
        is_fraud,
        latency,
    )

    return Prediction(
        is_fraud=is_fraud,
        probability=round(prob, 6),
        threshold=THRESHOLD,
        model=MODEL_NAME,
        latency_ms=round(latency, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}
