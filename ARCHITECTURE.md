# Architecture — Cross-Project Design Patterns

This document describes the recurring infrastructure patterns used across the fraud detection and GenAI agent projects. Written for technical interviewers evaluating production ML experience.

## Feature Engineering & Storage

Features are computed from BigQuery tables via scheduled SQL pipelines and materialised into a feature table partitioned by event date. The fraud pipeline processes 200M+ rows/month using parallelised Python scripts (multiprocessing + joblib) for metrics like Information Value (IV), Kolmogorov-Smirnov (KS), and mutual information. Feature freshness is bounded: real-time features (transaction amount, device fingerprint) are available at inference time; aggregate features (30-day velocity, frequency-encoded merchant ID) refresh every 6 hours via batch jobs on Databricks.

## Model Training & Registry

All experiments are tracked through MLflow: hyperparameters, dataset hashes, and evaluation metrics (ROC-AUC, Brier Score, ECE) are logged per run. The best model is promoted to the MLflow Model Registry with a `Staging` → `Production` transition gate. Models are exported as either scikit-learn joblib artifacts (for tree ensembles) or TorchScript/ONNX (for PyTorch models requiring mixed-precision inference). Retraining cadence is weekly for fraud models (concept drift driven by evolving fraud tactics) and monthly for the GenAI evaluation harness.

## Deployment & Rollout

Model images are containerised with Docker and deployed to GCP Vertex AI endpoints. The rollout follows a canary strategy: 1% of traffic is routed to the new model for 24 hours. During the canary window, the following signals are monitored:

| Signal | Threshold | Action if breached |
|--------|-----------|-------------------|
| `error_rate` | > 0.5% above baseline | Automated rollback |
| `latency_p50` | > 50ms | Alert, investigate |
| `latency_p95` | > 200ms | Automated rollback |
| `fraud_catch_rate` | < 5% drop from baseline | Alert, manual review |
| `false_positive_rate` | > 10% increase from baseline | Alert, manual review |

If no alert fires during the 24h window, traffic is ramped to 100%. Rollback is automated via Vertex AI traffic splitting — reverting to the previous model version takes < 60 seconds.

## Monitoring & Drift Detection

Production monitoring uses Prometheus for metric collection and Grafana dashboards for visualisation. Key monitoring layers:

1. **Infrastructure**: request latency (p50/p95), error rate, memory usage — standard SRE signals.
2. **Data drift**: PSI and KS test computed on incoming feature distributions vs. training reference. PSI ≥ 0.20 triggers a `HIGH` alert and automatic retraining ticket. PSI ∈ [0.10, 0.20) triggers `MEDIUM` (investigate).
3. **Model performance**: ground-truth labels arrive with a 30-day lag (chargeback dispute window). Weekly batch evaluation computes ROC-AUC, precision@recall=0.70, and ECE against the latest labelled cohort. A sustained ≥ 3% AUC drop over two consecutive weeks triggers retraining.
4. **Business KPIs**: fraud loss rate ($), false-positive rate (analyst workload). These are tracked in a separate BI dashboard and reviewed with stakeholders bi-weekly.

## Alert Escalation

```
PSI ≥ 0.20 (HIGH)   → Slack #ml-alerts → auto-create retraining ticket
PSI ≥ 0.10 (MEDIUM) → Slack #ml-alerts → data scientist investigates within 24h
AUC drop ≥ 3%        → PagerDuty → on-call ML engineer
Latency p95 > 200ms  → PagerDuty → on-call SRE
```

## GenAI Agent Safety

The ARGUS agent follows a safe-by-design pattern: all tool calls are confined to a sandboxed execution environment, LLM outputs are grounded in retrieved documents (RAG), and every response includes source citations. The evaluation pipeline measures faithfulness (fraction of claims supported by retrieved context) and citation accuracy (fraction of citations that map to actual source passages). There is no autonomous action on external systems — tool outputs are presented to the user for confirmation.
