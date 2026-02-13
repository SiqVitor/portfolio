# Public Projects

Selected projects by Vitor de Siqueira Rodrigues — Data Scientist / ML Engineer.

## Projects

### [End-to-End Fraud Detection](Fraud%20Detection/)
Production-grade fraud detection pipeline on the PaySim dataset (6.3 M mobile-money transactions). Time-based train/val/test split, balance-discrepancy feature engineering, five-model comparison (LR, RF, LightGBM, XGBoost, CatBoost) with `RandomizedSearchCV`, probability-calibration analysis (Brier Score, ECE, reliability curves), SHAP interpretability, hard-sample log-loss analysis, threshold optimisation, simulated data-drift monitoring (PSI, KS, Evidently AI), and a FastAPI serving endpoint.

### [ds_tools — Reusable ML Toolkit](ds_tools/)
Installable Python package (`pip install -e ds_tools/`) with sklearn-compatible transformers (`FrequencyEncoder`, `OutlierClipper`, `BalanceDeltaTransformer`), a one-call `ClassificationEvaluator` (ROC, PR, calibration, confusion matrix, hard-sample analysis), SHAP visualisation helpers, and data-drift detection utilities (PSI, KS test, simulated drift, multi-feature drift reports).

### [End-to-End GenAI Agent (ARGUS)](End%20to%20End%20GenAI%20Agent/)
Multi-agent system built with LangGraph for automated research, data analysis, and document-grounded Q&A. Features hybrid RAG retrieval, streaming responses, and an automated evaluation pipeline. Stack: Python, LangGraph, LangChain, FastAPI, ChromaDB, PostgreSQL, Redis, Streamlit.

### [Kaggle — House Prices: Advanced Regression Techniques](Hands%20On%20Statistics%20and%20Kaggle/)
Tabular regression on the Ames Housing dataset. Log-target transform, `InteractionFeatures` via sklearn `Pipeline` + `ColumnTransformer` for leak-free preprocessing. Six-model cross-validated comparison (Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR) with `RandomizedSearchCV` tuning and a `StackingRegressor` ensemble. Kaggle RMSLE **0.12279** — top 12.5% on the leaderboard.

### [Kaggle — Titanic: Machine Learning from Disaster](Hands%20On%20Statistics%20and%20Kaggle/)
Binary classification with domain-driven feature engineering (title extraction, family size, cabin deck, fare-per-person, ticket frequency). Six-model baseline (Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM), `GridSearchCV` hyperparameter tuning, and `VotingClassifier` / `StackingClassifier` ensembles. Best CV accuracy **0.8574** (Tuned GBC).

### [Practical Statistics for Data Scientists](Hands%20On%20Statistics%20and%20Kaggle/)
Applied exercises from the O'Reilly book (Bruce, Bruce & Gedeck). Covers estimates of location and variability, weighted metrics, percentile analysis, and distributional diagnostics with annotated visualizations.
