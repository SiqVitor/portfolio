# End-to-End Fraud Detection — IEEE-CIS E-Commerce Transactions

Production-grade fraud detection pipeline built on the **IEEE-CIS Fraud Detection** dataset — real-world e-commerce transaction data from **Vesta Corporation** (~590 k transactions, 434 features including Vesta-engineered signals and device/identity data).

The project showcases the full ML lifecycle — from exploratory analysis to multi-model comparison (including a **PyTorch neural network with entity embeddings**), probability calibration, SHAP-based interpretability, and simulated production monitoring with data-drift alerts.

---

## Project Roadmap

| Phase | Notebook | Key Deliverables |
|-------|----------|-----------------|
| **1 — Discovery** | `01_eda_and_roadmap.ipynb` | Business context, missing-value structure (V-column groups), class distribution (~3.5 % fraud), fraud by ProductCD/card/email, temporal patterns |
| **2 — Modeling** | `02_modeling_and_evaluation.ipynb` | Feature engineering (leakage-free aggregations), 5-model tree comparison + **PyTorch NN** (entity embeddings, AMP, OneCycleLR, early stopping), `RandomizedSearchCV`, calibration (Brier, ECE), SHAP, hard-sample analysis, threshold optimisation |
| **3 — Monitoring** | `03_monitoring_and_drift.ipynb` | Time-batch performance tracking, PSI & KS drift detection, simulated concept drift, Evidently AI reports, alert-threshold design |
| **4 — Serving** | `serve/app.py` | FastAPI inference endpoint with feature-pipeline architecture |

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROC-AUC | > 0.95 | Realistic target for real-world e-commerce data (harder than synthetic datasets) |
| Brier Score | < 0.03 | Probabilities must be calibrated — a 70 % prediction should be correct ~70 % of the time |
| ECE | < 0.05 | Quantifies calibration gap across probability bins |
| Precision @ Recall 0.70 | > 0.50 | Business constraint: balance fraud catch rate vs false-alarm cost |

---

## Setup

```bash
# 1. Install the reusable toolkit (from repo root)
pip install -e ds_tools/

# 2. Install project dependencies
pip install -r "Fraud Detection/requirements.txt"

# 3. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Configure Kaggle credentials (one-time)
#    Create ~/.kaggle/kaggle.json with your API token

# 5. Run notebooks in order (01 → 02 → 03)
```

---

## Dataset

**IEEE-CIS Fraud Detection** (`ieee-fraud-detection` on Kaggle) — real e-commerce transaction data provided by Vesta Corporation.

| Property | Value |
|----------|-------|
| Transactions | ~590,540 |
| Features | 394 (transaction) + 41 (identity) |
| Fraud cases | ~20,663 (~3.5 %) |
| Identity coverage | ~25 % of transactions have device/browser data |
| Key feature groups | TransactionAmt, card1–card6, addr, C1–C14, D1–D15, M1–M9, V1–V339, DeviceType |

The realistic class imbalance, massive structured missingness (V columns), and high dimensionality make this a challenging benchmark that reflects real production fraud systems.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, NumPy, kagglehub |
| Modelling | scikit-learn, LightGBM, XGBoost, CatBoost |
| Deep Learning | **PyTorch** (entity embeddings, AMP, OneCycleLR) |
| Calibration | `ds_tools.evaluation` (Brier, ECE, reliability curves) |
| Interpretability | SHAP |
| Monitoring | `ds_tools.monitoring` (PSI, KS), Evidently AI |
| Serving | FastAPI, joblib |
| Visualisation | matplotlib, seaborn |
