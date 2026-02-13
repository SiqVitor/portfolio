# End-to-End Fraud Detection — Mobile Money Transactions

Production-grade fraud detection pipeline built on the **PaySim** synthetic dataset (6.3 M transactions, 31-day simulation based on real mobile-money logs from an African country).

The project showcases the full lifecycle — from exploratory analysis to model calibration, SHAP-based interpretability, and simulated production monitoring with data-drift alerts.

---

## Project Roadmap

| Phase | Notebook | Key Deliverables |
|-------|----------|-----------------|
| **1 — Discovery** | `01_eda_and_roadmap.ipynb` | Business context, class distribution, fraud-by-type analysis, balance discrepancy patterns, temporal heatmaps |
| **2 — Modeling** | `02_modeling_and_evaluation.ipynb` | Feature engineering, 5-model comparison (LR, RF, LightGBM, XGBoost, CatBoost), `RandomizedSearchCV`, calibration analysis (Brier, ECE), SHAP, hard-sample analysis, threshold optimisation |
| **3 — Monitoring** | `03_monitoring_and_drift.ipynb` | Time-batch performance tracking, PSI & KS drift detection, simulated concept drift, Evidently AI reports, alert-threshold design |
| **4 — Serving** | `serve/app.py` | FastAPI inference endpoint with latency logging |

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROC-AUC | > 0.99 | Strong signal in balance discrepancies makes high AUC achievable |
| Brier Score | < 0.01 | Probabilities must be *honest* — a 70 % prediction should be correct ~70 % of the time |
| ECE | < 0.02 | Quantifies calibration gap across probability bins |
| Precision @ Recall 0.80 | > 0.90 | Business constraint: minimise false alarms without missing real fraud |

---

## Setup

```bash
# 1. Install the reusable toolkit (from repo root)
pip install -e ds_tools/

# 2. Install project dependencies
pip install -r "Fraud Detection/requirements.txt"

# 3. Configure Kaggle credentials (one-time)
#    Create ~/.kaggle/kaggle.json with your API token
#    Or set KAGGLE_USERNAME / KAGGLE_KEY env vars

# 4. Run notebooks in order (01 → 02 → 03)
```

---

## Dataset

**PaySim** (`ealaxi/paysim1` on Kaggle) — a multi-agent simulation calibrated with real aggregate statistics from a mobile-money provider.

| Property | Value |
|----------|-------|
| Transactions | 6,362,620 |
| Features | 11 (step, type, amount, balances, flags) |
| Fraud cases | ~8,213 (0.13 %) |
| Fraud types | TRANSFER and CASH_OUT only |
| Time span | 744 hours (31 days) |

The severe class imbalance (0.13 %) and the restriction of fraud to only two transaction types make this a realistic benchmark for production fraud systems.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, NumPy |
| Modelling | scikit-learn, LightGBM, XGBoost, CatBoost |
| Calibration | `ds_tools.evaluation` (Brier, ECE, reliability curves) |
| Interpretability | SHAP |
| Monitoring | `ds_tools.monitoring` (PSI, KS), Evidently AI |
| Serving | FastAPI, joblib |
| Visualisation | matplotlib, seaborn |

All tools are **free and open-source**.
