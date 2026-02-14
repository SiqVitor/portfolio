# ML Platform — End-to-End Lifecycle Orchestration

Demonstrates ML lifecycle management: experiment tracking, data validation, pipeline orchestration, model registry, and reproducibility. Modeled after internal ML platforms at organisations with multiple models in production.

## Objective

Provide a repeatable, auditable pipeline that takes raw data through validation → feature engineering → training → evaluation → registration. Every run produces a traceable artifact (parameters, metrics, model file, data hash) enabling reproducibility and governance.

This is a portfolio demonstration, not a deployed platform. The patterns reflect experience managing the ML lifecycle across BigQuery/Databricks ingestion, MLflow tracking and model registry, CI/CD for model images, and production monitoring at Mercado Livre (CV: "Owned the ML lifecycle: data ingestion, MLflow experiment tracking and model registry, CI/CD for model images, production monitoring").

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA VALIDATION                               │
│  Schema checks  →  Distribution checks  →  Missing value audit   │
│  Fail-fast if data contract is violated                          │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                               │
│  Deterministic transforms (logged parameters)                    │
│  FrequencyEncoder, OutlierClipper (ds_tools)                     │
│  Output: feature matrix + feature metadata                       │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
│  Experiment tracking (MLflow-style local logging)                │
│  Hyperparameters, dataset hash, split strategy → all logged      │
│  Multiple model comparison (LightGBM, Logistic Regression)       │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EVALUATION GATE                               │
│  Metric thresholds: AUC > 0.90, Brier < 0.05, ECE < 0.08       │
│  Champion-challenger comparison                                  │
│  Pass → register; Fail → alert, do not deploy                   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MODEL REGISTRY                                 │
│  Versioned model artifact (joblib + metadata JSON)               │
│  Stages: Development → Staging → Production                     │
│  Lineage: data hash + params + metrics → model version           │
└──────────────────────────────────────────────────────────────────┘
```

## Metrics

| Type | Metric | Description |
|------|--------|-------------|
| **Business** | Fraud loss reduction ($), false positive rate | Downstream impact of model quality |
| **ML** | ROC-AUC, Brier Score, ECE, Log Loss | Model quality gates |
| **Operational** | Pipeline duration, validation pass rate, registry write success | Platform health |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Pipeline | Python (lightweight sequential orchestration) |
| Experiment tracking | Local file-based logging (mirrors MLflow tracking API) |
| Model registry | File-based versioned artifacts (mirrors MLflow model registry) |
| Data validation | Schema + distribution checks (custom, minimal) |
| Training | LightGBM, scikit-learn LogisticRegression |
| Evaluation | ds_tools (ClassificationEvaluator, Brier, ECE) |

## Run the Demo

```bash
bash ml_platform/demo/run_pipeline.sh
```

**What happens:**
1. Generates synthetic dataset (10,000 transactions)
2. Validates data schema and distributions
3. Engineers features (frequency encoding, outlier clipping)
4. Trains two models (LightGBM, Logistic Regression)
5. Evaluates both, selects champion by AUC
6. Registers the winning model with full metadata

**Expected output in `ml_platform/demo/results/`:**
- `metrics.json` — evaluation metrics for both models + champion selection
- `model_registry/` — versioned model artifact + metadata
- `validation_report.json` — data quality checks

## Related Files

- [Architecture decisions](architecture.md) — experiment lifecycle, governance, retraining triggers
- [Monitoring strategy](monitoring.md) — pipeline health, model quality, business impact
- [ds_tools](../ds_tools/) — reusable evaluation and preprocessing toolkit
