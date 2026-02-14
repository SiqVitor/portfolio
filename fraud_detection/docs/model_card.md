# Model Card — Fraud Detection (LightGBM)

## Model Details

| Field | Value |
|-------|-------|
| Model name | Fraud Detection — LightGBM Classifier |
| Version | Demo v1.0 |
| Type | LightGBM binary classifier |
| Owner | Vitor Rodrigues |
| Date trained | Demo: on-demand; Production: [MISSING: confirm date] |
| Framework | LightGBM 4.x, scikit-learn 1.3+ |
| License | MIT (demo); Proprietary (production) |

## Intended Use

- **Primary use**: Binary fraud classification on e-commerce transactions (chargeback and ATO detection)
- **Intended users**: Fraud operations team, automated transaction scoring pipeline
- **Out-of-scope uses**: Credit scoring, identity verification, or any use outside transaction-level fraud classification

## Training Data

| Property | Value |
|----------|-------|
| Public dataset | IEEE-CIS Fraud Detection (Vesta Corporation) — 590,540 transactions, 434 features |
| Production dataset | Proprietary (Mercado Livre) — not publicly available |
| Demo dataset | Synthetic — 10,000 transactions via `demo/generate_synthetic.py` |
| Label distribution | ~3.5% fraud (IEEE-CIS); ~5% fraud (synthetic demo) |
| Time range | [MISSING: confirm production date range] |
| Split strategy | Time-based: train → validation → test (no future leakage) |

## Evaluation Metrics

| Metric | Demo Value | Target | Production |
|--------|-----------|--------|------------|
| ROC-AUC | See `results/summary.json` | > 0.95 | [MISSING: confirm] |
| Precision @ Recall=0.70 | See `results/summary.json` | > 0.50 | [MISSING: confirm] |
| Brier Score | See `results/summary.json` | < 0.03 | [MISSING: confirm] |
| ECE | See `results/summary.json` | < 0.05 | [MISSING: confirm] |
| Log Loss | See `results/summary.json` | — | [MISSING: confirm] |

## Ethical Considerations

- **Fairness**: [MISSING: describe demographic groups tested for bias]
- **Privacy**: Transaction features are anonymised; no PII in public datasets. Production data handled under Mercado Livre data governance policies.
- **Feedback loops**: Model decisions influence which transactions are reviewed, which generates labels for future training. Mitigated by random sampling of un-flagged transactions for labelling.

## Limitations

- Demo model trained on synthetic data — not representative of production performance
- Production model performance degrades on transaction types not seen in training (e.g., new product categories, new payment methods)
- Ground-truth labels have a 30-day lag (chargeback dispute window)
- Feature freshness: aggregate features (velocity, frequency encoding) refresh every 6 hours — fraud patterns within that window may be missed

## Monitoring & Maintenance

| Aspect | Details |
|--------|---------|
| Retraining cadence | Weekly (production); on-demand (demo) |
| Drift detection | PSI + KS test on top 20 features; PSI ≥ 0.20 → retrain |
| Performance tracking | Weekly batch eval on labelled cohort (30-day label lag) |
| Rollback procedure | Vertex AI traffic split; revert to previous version in < 60s |
