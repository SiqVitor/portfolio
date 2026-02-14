# Real-Time ML System — Online Inference Pipeline

Demonstrates the design and implementation of an online inference pipeline with streaming features, batch-trained models, and latency-aware serving. Modeled after real-time fraud scoring systems at scale.

## Objective

Score incoming transactions in real-time (< 50ms p95) using a model trained offline on historical data, while incorporating streaming features that update as events arrive.

This is a portfolio demonstration — not a claim of a specific production deployment. The architecture patterns reflect experience designing fraud prevention systems processing 200M+ rows/month at Mercado Livre.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      EVENT STREAM                                │
│  Kafka / simulated queue  →  Transaction events (JSON)           │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FEATURE ASSEMBLY                               │
│  Real-time features      +  Pre-computed features (feature store)│
│  (from event payload)       (batch-refreshed every 6h)           │
│  amount, device, IP         velocity, freq encoding, aggregates  │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ONLINE INFERENCE                               │
│  Pre-loaded model (LightGBM joblib)                              │
│  Single-event scoring: feature vector → P(fraud)                 │
│  Latency target: p50 < 10ms, p95 < 50ms                         │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                   DECISION + LOGGING                             │
│  Threshold-based action (block / review / approve)               │
│  Log prediction + features + latency for monitoring              │
└──────────────────────────────────────────────────────────────────┘
```

## Metrics

| Type | Metric | Target |
|------|--------|--------|
| **Business** | Fraud loss rate ($) | Minimize — prevented chargebacks |
| **ML** | ROC-AUC, Precision@Recall=0.70, Brier Score | > 0.95, > 0.50, < 0.03 |
| **Operational** | Latency p50/p95, throughput (events/sec), error rate | < 10ms / < 50ms, > 1000/sec, < 0.1% |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Streaming | Kafka (simulated via in-process queue in demo) |
| Feature store | In-memory dict (simulates Redis/Feast lookup) |
| Model serving | Pre-loaded LightGBM (joblib) |
| Monitoring | Latency histograms, prediction distribution tracking |
| Language | Python 3.10+ |

## Run the Demo

```bash
bash realtime_ml_system/demo/run_demo.sh
```

**What happens:**
1. Trains a LightGBM model on synthetic data (batch training phase)
2. Simulates 1,000 streaming events through the online inference pipeline
3. Scores each event in real-time with latency tracking
4. Outputs `results/summary.json` with metrics and latency percentiles

**Expected output:**
```json
{
  "events_processed": 1000,
  "fraud_detected": 48,
  "latency_p50_ms": 0.12,
  "latency_p95_ms": 0.35,
  "latency_p99_ms": 0.52,
  "throughput_events_per_sec": 3200,
  "model_roc_auc": 0.998,
  "avg_prediction": 0.047
}
```
(Exact values will vary by hardware.)

## Related Files

- [Architecture decisions](docs/architecture.md) — online vs batch, feature freshness, failure modes
- [Monitoring strategy](docs/monitoring.md) — signals, thresholds, alert escalation
- [Fraud case study](../fraud_detection/) — end-to-end fraud detection pipeline
