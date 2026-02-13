# ds_tools

Reusable Data Science & Machine Learning toolkit designed for production-grade ML projects.

## Installation

```bash
# From the repository root
pip install -e ds_tools/

# With monitoring extras (Evidently AI)
pip install -e "ds_tools/[monitoring]"
```

## Package Structure

```
ds_tools/
├── evaluation/
│   ├── calibration.py   — Brier Score, ECE, reliability curves
│   └── report.py        — ClassificationEvaluator (ROC, PR, confusion, calibration)
├── preprocessing/
│   └── transformers.py  — Sklearn-compatible transformers (FrequencyEncoder, OutlierClipper, etc.)
├── visualization/
│   └── plots.py         — SHAP summaries, ROC-PR overlays, threshold analysis
└── monitoring/
    └── drift.py         — PSI, KS test, simulated drift, drift reports
```

## Quick Start

```python
from ds_tools.evaluation import ClassificationEvaluator
from ds_tools.evaluation.calibration import expected_calibration_error
from ds_tools.preprocessing.transformers import FrequencyEncoder
from ds_tools.monitoring.drift import psi, drift_report

# Full evaluation in 3 lines
evaluator = ClassificationEvaluator(y_true, y_prob, model_name="LightGBM")
evaluator.summary()
evaluator.plot_full_report()
```

## Design Principles

- **Sklearn-compatible**: All transformers inherit from `BaseEstimator` + `TransformerMixin` and work inside `Pipeline`.
- **Production-oriented**: Functions are stateless where possible; fitted objects are serialisable via `joblib`.
- **Opinionated defaults**: One call gives you BrierScore + ECE + Calibration Curve + ROC + PR + Confusion Matrix.
