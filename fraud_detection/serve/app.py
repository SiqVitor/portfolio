"""Fraud Detection — FastAPI inference endpoint (IEEE-CIS).

Architecture note:
    In production, a separate feature pipeline (Airflow / Spark) builds
    the feature vector.  This endpoint receives the pre-computed feature
    dict and runs inference — clean separation of concerns.

Run:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000

Request:
    POST /predict
    {
        "features": {
            "TransactionAmt": 150.0,
            "log_amt": 5.017,
            "card1": 12345,
            ...
        }
    }
"""

from __future__ import annotations

import logging
import os
import time

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
    os.path.dirname(__file__), "..", "demo", "results", "fraud_model.joblib"
)
artefacts = joblib.load(ARTEFACT_PATH)

MODEL = artefacts["model"]
FEATURE_COLS = artefacts["feature_cols"]
THRESHOLD = artefacts["threshold"]
MODEL_NAME = artefacts["model_name"]
TRAIN_MEDIANS = pd.Series(artefacts["train_medians"])
NUMERIC_COLS = artefacts["numeric_cols"]

logger.info(
    "Loaded %s  |  threshold=%.4f  |  %d features",
    MODEL_NAME,
    THRESHOLD,
    len(FEATURE_COLS),
)

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="Fraud Detection API", version="0.2.0")


class PredictionRequest(BaseModel):
    """Input: pre-computed feature vector as a dict.

    In a real setup, an upstream feature pipeline (Spark, Airflow)
    builds this dict from the raw transaction + identity data.
    """

    features: dict


class Prediction(BaseModel):
    is_fraud: bool
    probability: float
    threshold: float
    model: str
    latency_ms: float


@app.post("/predict", response_model=Prediction)
def predict(req: PredictionRequest) -> Prediction:
    t0 = time.perf_counter()

    # Optimized inference: construct numpy array directly (avoiding pandas overhead)
    x_input = []
    for col in FEATURE_COLS:
        val = req.features.get(col)
        # Impute missing values with training medians if available
        if val is None:
            val = TRAIN_MEDIANS.get(col, np.nan)
        x_input.append(val)

    x = np.array(x_input).reshape(1, -1)
    prob = float(MODEL.predict_proba(x)[:, 1][0])
    is_fraud = prob >= THRESHOLD

    latency = (time.perf_counter() - t0) * 1000
    logger.info(
        "prob=%.4f  fraud=%s  latency=%.1fms",
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
    return {"status": "ok", "model": MODEL_NAME, "features": len(FEATURE_COLS)}
