# Monitoring Strategy — Real-Time ML System

Three monitoring layers: infrastructure, model quality, and business impact.

## Layer 1 — Infrastructure (real-time)

Collected via Prometheus, visualised in Grafana.

| Signal | Collection | Alert threshold | Escalation |
|--------|-----------|-----------------|------------|
| `inference_latency_p50` | Histogram per request | > 10ms sustained 5min | Slack #ml-alerts |
| `inference_latency_p95` | Histogram per request | > 50ms sustained 5min | PagerDuty on-call |
| `inference_error_rate` | Counter (errors / total) | > 0.1% sustained 5min | PagerDuty on-call |
| `throughput_rps` | Counter per second | < 500 events/sec (below expected baseline) | Slack #ml-alerts |
| `feature_store_latency` | Histogram per lookup | > 5ms p95 | Slack #ml-alerts |
| `feature_store_miss_rate` | Counter (misses / lookups) | > 5% | Slack #ml-alerts |
| `memory_usage_mb` | Gauge per worker | > 80% of limit | Slack #infra |

## Layer 2 — Model Quality (daily + weekly)

| Signal | Frequency | Alert threshold | Action |
|--------|-----------|-----------------|--------|
| PSI on top 20 input features | Daily batch | PSI ≥ 0.20 (HIGH) | Auto-create retraining ticket |
| PSI moderate | Daily batch | 0.10 ≤ PSI < 0.20 | Investigate within 24h |
| KS test on prediction distribution | Daily batch | p-value < 0.01 | Investigate — possible concept drift |
| ROC-AUC on labelled cohort | Weekly batch (30-day lag) | Drop ≥ 3% for 2 consecutive weeks | Trigger retraining |
| Brier Score / ECE | Weekly batch | Brier > 0.05 or ECE > 0.08 | Investigate calibration |
| Prediction distribution mean | Daily | Shift > 20% from baseline | Investigate |

## Layer 3 — Business Impact (weekly)

| Signal | Frequency | Owner | Notes |
|--------|-----------|-------|-------|
| Fraud loss rate ($) | Weekly | Risk ops | Monetary loss from undetected fraud |
| False positive rate | Weekly | Risk ops | Analyst workload from false alarms |
| Manual review volume | Weekly | Risk ops | Should decrease with model improvement |
| Chargeback rate | Monthly (30-day lag) | Finance | Ground-truth for model performance |

## Degraded Mode Indicators

| Condition | Response |
|-----------|----------|
| Feature store returning empty features > 5% | Switch to real-time-only feature vector; alert data engineering |
| Model prediction latency > 100ms | Investigate model size / memory pressure; consider model pruning |
| All predictions clustered near 0.5 | Model likely producing random scores — possible data pipeline issue |
| Prediction volume drops > 50% from baseline | Upstream pipeline failure — escalate to data engineering |

## Dashboard Layout

```
Row 1: [Throughput (events/sec)]  [Latency p50/p95/p99]  [Error Rate]
Row 2: [Prediction Distribution]  [Feature Store Hit Rate]  [Memory]
Row 3: [PSI Heatmap (features)]  [Weekly AUC Trend]  [Business Loss $]
```
