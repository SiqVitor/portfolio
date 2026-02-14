# Architecture Decisions — ML Platform

Design rationale for the ML lifecycle orchestration platform.

## Experiment Lifecycle

```
Code change → Run pipeline → Log experiment → Compare → Promote → Deploy
                                  │
                                  ▼
                          Experiment Store
                          (params, metrics,
                           data hash, model)
```

### Experiment Tracking

Every pipeline run logs:
- **Parameters**: hyperparameters, feature list, split ratio, random seed
- **Metrics**: ROC-AUC, Brier Score, ECE, Log Loss, precision, recall
- **Data lineage**: SHA-256 hash of the training dataset, row count, label distribution
- **Artifacts**: serialised model (joblib), feature importance, calibration curve

In production, this maps to MLflow's tracking API. The demo uses file-based logging with the same structure (JSON metadata + artifact directory).

### Champion-Challenger

When a new model is trained:
1. Evaluate on the same held-out test set as the current champion
2. Compare primary metric (ROC-AUC) — new model must meet or exceed champion
3. Compare secondary metrics (Brier, ECE) — must not regress by > 10% relative
4. If both pass → promote to Staging → canary deployment
5. If fail → log result, keep current champion, alert data scientist

## Governance Considerations

| Concern | Implementation |
|---------|---------------|
| **Reproducibility** | Fixed random seeds, pinned dependency versions, data hash logged per run |
| **Auditability** | Every model version linked to: training data hash, parameters, evaluation metrics, who triggered the run |
| **Approval gates** | Model promotion from Development → Staging requires metric thresholds. Staging → Production requires 24h canary pass. |
| **Data privacy** | Training data remains in the data platform (BigQuery/Databricks). Only model artifacts are exported. No PII in model files. |
| **Model retirement** | Models older than 90 days without retraining are flagged for review. Models with sustained AUC degradation are automatically moved to Archived. |

## Retraining Triggers

| Trigger | Source | Action |
|---------|--------|--------|
| **Scheduled** | Cron (weekly) | Full pipeline: validate → train → evaluate → register |
| **Drift alert** | Monitoring (PSI ≥ 0.20) | Trigger retraining with fresh data window |
| **Performance degradation** | Weekly batch eval (AUC drop ≥ 3% for 2 weeks) | Trigger retraining + alert to data scientist |
| **Manual request** | Data scientist | Ad-hoc pipeline run with custom parameters |
| **Data contract change** | Schema migration | Full pipeline run + validation of new schema |

## CI/CD Integration

```
git push (model code change)
    │
    ▼
CI: lint + unit tests + smoke pipeline (synthetic data)
    │
    ▼
CD: full pipeline on staging data → evaluate → register to Staging
    │
    ▼
Manual approval → promote to Production → canary deploy
```

### Pipeline as Code

The training pipeline is a Python script (not a notebook). This enables:
- Version control (git diff on pipeline changes)
- CI integration (run on every PR)
- Parameterisation (different configs for dev, staging, production)
- Testing (unit tests on validation and feature engineering functions)

## Versioning Strategy

| Artifact | Versioning | Storage |
|----------|-----------|---------|
| Training code | git (commit SHA) | GitHub |
| Training data | SHA-256 hash + row count | Logged in experiment metadata |
| Model artifact | Auto-increment integer (v1, v2, ...) | Model registry (file-based demo / MLflow production) |
| Feature schema | Semantic versioning | Committed alongside training code |
| Pipeline config | git (same repo as training code) | GitHub |

**Model version metadata** includes:
```json
{
  "version": 3,
  "stage": "staging",
  "created_at": "2025-01-15T10:30:00Z",
  "data_hash": "sha256:abc123...",
  "training_rows": 100000,
  "metrics": {"roc_auc": 0.965, "brier": 0.021},
  "params": {"n_estimators": 200, "max_depth": 5},
  "git_commit": "abc1234"
}
```
