# Architecture Decisions — Fraud Detection

Trade-off analysis for key design decisions in the fraud detection pipeline.

## Online vs. Batch Scoring

| Dimension | Online (real-time) | Batch |
|-----------|-------------------|-------|
| Latency | < 100ms per request | Minutes–hours |
| Use case | Transaction approval/block at checkout | Retroactive review, labelling |
| Feature availability | Only real-time features (amount, device, IP) | Full aggregate features (velocity, frequency) |
| Infra cost | Persistent endpoint (Vertex AI) | Scheduled jobs (Databricks) |

**Choice**: Hybrid. Real-time scoring at transaction time using a subset of features for the approve/block decision. Batch re-scoring every 6 hours with full aggregate features for retroactive review and model performance evaluation.

**Why**: Pure real-time scoring misses aggregate signals (e.g., 30-day velocity) that are among the strongest predictors. Pure batch misses the fraud prevention window entirely.

## Latency SLOs

| Metric | Target |
|--------|--------|
| p50 | < 50ms |
| p95 | < 200ms |
| p99 | < 500ms |

Monitored via Prometheus; p95 breach triggers PagerDuty alert.

## Feature Freshness

| Feature Type | Refresh | Examples |
|-------------|---------|----------|
| Real-time | At inference | Transaction amount, device type, IP geolocation |
| Near-real-time (1h) | Streaming aggregation | Hourly transaction count per card |
| Batch (6h) | Scheduled pipeline | 30-day velocity, frequency-encoded merchant, IV/KS metrics |

## Retraining Cadence

Weekly, triggered by either:
1. Calendar schedule (every Monday)
2. Drift alert (PSI ≥ 0.20 on any top-20 feature)
3. Performance degradation (AUC drop ≥ 3% for two consecutive weekly evals)

Retraining uses the most recent 90 days of labelled data. The new model is promoted to `Staging` in MLflow, evaluated against the current `Production` model on a held-out week, and promoted only if it meets or exceeds all metric targets.

---

## Decision Log

1. **LightGBM as primary model over deep learning for production scoring** — LightGBM provides comparable AUC to the PyTorch entity-embedding model on this feature set, with 10x lower inference latency (< 5ms vs. ~50ms) and simpler deployment (joblib vs. TorchScript). PyTorch is used for research/experimentation and as a secondary model for ensemble scoring in the batch pipeline.

2. **Time-based splits over random splits** — Random splitting leaks future information into training (e.g., a transaction from next month could appear in the training set). Time-based splits simulate production conditions where only past data is available at prediction time. This gives more honest AUC/Brier estimates.

3. **Calibration (Brier/ECE) as a first-class metric alongside AUC** — Downstream business rules apply dollar-dependent thresholds (e.g., "block if P(fraud) > 0.7 and amount > $500"). If probabilities are uncalibrated, these thresholds produce unpredictable behaviour. Brier Score and ECE are tracked in MLflow and included in the promotion gate.

4. **Canary rollout over blue-green deployment** — Blue-green gives a binary switch but no gradual risk mitigation. Canary (1% traffic for 24h) exposes the new model to real traffic at low risk, with automated rollback if `error_rate` or `latency_p95` exceeds thresholds. The trade-off is 24h slower full deployment, which is acceptable for a model updated weekly.
