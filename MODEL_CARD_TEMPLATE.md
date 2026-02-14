# Model Card — [Model Name]

## Model Details

| Field | Value |
|-------|-------|
| Model name | [Name] |
| Version | [e.g., 1.0.0] |
| Type | [e.g., LightGBM classifier, PyTorch NN] |
| Owner | [Team / individual] |
| Date trained | [YYYY-MM-DD] |
| Framework | [e.g., scikit-learn 1.3, PyTorch 2.x] |
| License | [e.g., Proprietary, MIT] |

## Intended Use

- **Primary use**: [What the model does, e.g., "binary fraud classification on e-commerce transactions"]
- **Intended users**: [e.g., fraud operations team, automated decision pipeline]
- **Out-of-scope uses**: [What the model should NOT be used for]

## Training Data

| Property | Value |
|----------|-------|
| Dataset | [Name, source, or "proprietary — not publicly available"] |
| Size | [rows × features] |
| Label distribution | [e.g., 3.5% positive / 96.5% negative] |
| Time range | [e.g., Jan 2023 – Dec 2023] |
| Split strategy | [e.g., time-based: train < cutoff < val < cutoff < test] |

## Evaluation Metrics

| Metric | Value | Target |
|--------|-------|--------|
| ROC-AUC | [value] | [target] |
| Precision @ Recall=0.70 | [value] | [target] |
| Brier Score | [value] | [target] |
| ECE | [value] | [target] |
| Log Loss | [value] | — |

## Ethical Considerations

- **Fairness**: [Has the model been tested for bias across demographic groups? Which groups?]
- **Privacy**: [Does the model use PII? How is it protected?]
- **Feedback loops**: [Can the model's decisions affect future training data?]

## Limitations

- [Known failure modes, e.g., "performance degrades on transaction types not seen in training"]
- [Data freshness constraints]
- [Geographical or temporal scope limitations]

## Monitoring & Maintenance

| Aspect | Details |
|--------|---------|
| Retraining cadence | [e.g., weekly] |
| Drift detection | [e.g., PSI + KS on top 20 features, threshold PSI ≥ 0.20] |
| Performance tracking | [e.g., weekly batch eval on labelled cohort, 30-day label lag] |
| Rollback procedure | [e.g., Vertex AI traffic split, < 60s revert] |

## Caveats & Recommendations

- [Any additional context for consumers of this model]
