# Monitoring Strategy — ML Platform

Monitoring for the ML lifecycle platform covers pipeline health, model quality, and downstream business impact.

## Layer 1 — Pipeline Health (per-run)

| Signal | Source | Alert threshold |
|--------|--------|-----------------|
| Pipeline duration | Pipeline run metadata | > 2x median duration |
| Data validation pass/fail | Validation step | Any FAIL → block pipeline, alert |
| Training convergence | Loss curve | No improvement in last 20% of iterations |
| Registry write success | Model registration step | Failure → alert, retry once |
| Feature count mismatch | Feature engineering output | Expected N features, got M → block |

## Layer 2 — Model Quality (post-training)

| Signal | Evaluation | Alert threshold |
|--------|-----------|-----------------|
| ROC-AUC | Test set evaluation | < 0.90 (minimum threshold) |
| Brier Score | Test set evaluation | > 0.05 |
| ECE | Test set evaluation | > 0.08 |
| Champion-challenger delta | Comparison with current production model | New model AUC < champion AUC → do not promote |
| Feature importance shift | Compared to previous run | Top-5 feature order change → investigate |

## Layer 3 — Production Model (deployed)

| Signal | Frequency | Alert threshold | Action |
|--------|-----------|-----------------|--------|
| Prediction distribution mean | Daily | Shift > 20% from training baseline | Investigate data pipeline |
| PSI on input features | Daily | ≥ 0.20 → HIGH | Auto-trigger retraining |
| Labelled cohort AUC | Weekly (30-day label lag) | Drop ≥ 3% for 2 consecutive weeks | Trigger retraining |
| Serving latency | Real-time (if serving online) | p95 > 50ms | SRE investigation |
| Error rate | Real-time | > 0.1% | PagerDuty |

## Layer 4 — Business Impact (weekly/monthly)

| Signal | Owner | Notes |
|--------|-------|-------|
| Fraud loss rate ($) | Risk ops | Primary business metric |
| False positive rate | Risk ops | Analyst workload proxy |
| Model intervention rate | Product | % of transactions scored above threshold |
| Retraining frequency | ML platform | Should be stable; spikes indicate instability |

## Alerting

```
Pipeline failure         → Slack #ml-platform → on-call data scientist
Validation FAIL          → Slack #ml-platform → pipeline blocked automatically
Model below threshold    → Slack #ml-alerts   → do not promote, investigate
PSI HIGH (≥ 0.20)       → Slack #ml-alerts   → auto-create retraining ticket
AUC drop ≥ 3% (2 weeks) → PagerDuty          → on-call ML engineer
```
