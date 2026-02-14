# Missing Items & Placeholders

This file lists every numeric claim, metric, or detail used in the portfolio that was **not explicitly stated** in `cv_vitor_rodrigues.pdf` or `Vitor Rodrigues _ LinkedIn.pdf`. Items marked `[MISSING]` in other documents point here.

## Items to Confirm

| Location | Placeholder | Suggested Value | Notes |
|----------|-------------|-----------------|-------|
| `Fraud Detection/MODEL_CARD.md` | Production ROC-AUC | ≥ 0.95 | Target stated in Fraud Detection README; actual production metric not in CV/LinkedIn |
| `Fraud Detection/MODEL_CARD.md` | Production Brier Score | < 0.03 | Same — target from project, not a reported result |
| `Fraud Detection/MODEL_CARD.md` | Production ECE | < 0.05 | Same |
| `Fraud Detection/MODEL_CARD.md` | Production Precision@Recall=0.70 | > 0.50 | Same |
| `Fraud Detection/MODEL_CARD.md` | Training data time range | [YYYY – YYYY] | Exact dates not in CV/LinkedIn |
| `Fraud Detection/MODEL_CARD.md` | Exact number of features used | [N] | CV/LinkedIn don't specify for production models |
| `Fraud Detection/MODEL_CARD.md` | Fairness testing details | [Describe groups tested] | Not mentioned in CV/LinkedIn |
| `ARCHITECTURE.md` | Alert threshold: AUC drop ≥ 3% | 3% | Reasonable default; exact production threshold not in documents |
| `ARCHITECTURE.md` | Alert threshold: latency p95 > 200ms | 200ms | Reasonable default; exact SLO not in documents |
| `ARCHITECTURE.md` | Retraining cadence: weekly | Weekly | CV mentions "continuous monitoring"; exact cadence not specified |
| `ARCHITECTURE.md` | Feature refresh: every 6 hours | 6h | Reasonable default; not in documents |
| `ARCHITECTURE.md` | Rollback time: < 60s | < 60s | Vertex AI capability; not explicitly claimed |

## Items Confirmed from Documents

All other numeric claims used in the portfolio are documented in the [Implementation Plan](.) verified facts table and sourced directly from the CV or LinkedIn PDF.

## What Was Not Invented

- No client names, revenue figures, or deployment counts were fabricated
- No certifications or publications were added beyond what appears in the documents
- The demo uses only synthetic data — no proprietary datasets are included
- Architecture details (monitoring thresholds, cadence values) are labelled as reasonable defaults where not sourced
