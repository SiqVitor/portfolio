# Interview Talking Points

12 concise bullets in context → action → result → metric format. All numbers are sourced from CV and LinkedIn.

---

1. **ATO agent automation** — Mercado Livre's ATO review process required 20+ analysts manually triaging reports. I designed a GenAI agent framework using Python and LangChain to automate classification and report generation. Reduced manual workload by 70%, freeing 15–20 analysts for higher-value investigation.

2. **Fraud prevention revenue impact** — Chargeback and ATO fraud were causing significant monthly losses at Mercado Livre. I engineered end-to-end fraud detection models (feature design through deployment) that intercept fraudulent transactions before completion. Generated $30k–$50k USD in monthly savings.

3. **Feature engineering at scale** — Fraud models required features computed over 200M+ rows/month of transaction data. I refactored the feature engineering pipelines using parallelised Python (multiprocessing, joblib) with rigorous feature validation. Reduced computation time and enabled weekly model retraining.

4. **Deep learning for fraud** — Tree ensembles alone missed entity-level patterns in high-cardinality categorical features. I trained PyTorch models with entity embeddings, custom training loops, mixed precision (AMP), and DistributedDataParallel (DDP) for multi-GPU scaling. Exported production models via TorchScript/ONNX.

5. **Canary deployment and rollback** — Deploying new fraud models carries risk of increased false positives or missed fraud. I implemented canary rollout (1% traffic for 24h) on GCP Vertex AI with automated rollback triggered by error_rate and latency_p95 thresholds. Zero-downtime deployments with < 60s rollback.

6. **MLOps lifecycle ownership** — Scattered experiment tracking and manual deployments slowed the team's iteration speed. I established the full ML lifecycle: BigQuery/Databricks ingestion → MLflow tracking and model registry → CI/CD for model images → Prometheus/Grafana monitoring → drift detection.

7. **KPI monitoring automation (ClearSale)** — Contractual SLA breaches were detected too late due to manual dashboard checks. I built automated monitoring dashboards with proactive alert systems for risk metrics. Reduced KPI fluctuation response time by 40%.

8. **Cross-functional ML delivery (ClearSale)** — Enterprise e-commerce clients required fraud scoring models aligned to client-specific SLAs. I delivered end-to-end risk scoring projects: data sampling, feature engineering with production freshness constraints, A/B/shadow testing, and inference pipeline deployment. Consistently exceeded performance targets.

9. **Data visibility at Cappta** — The risk management department lacked consolidated reporting on transactional behaviour. I built SQL-based reporting architectures analysing payment patterns and fraud indicators. Improved data visibility by 50% for the risk team.

10. **Probability calibration expertise** — Standard classification metrics (accuracy, AUC) don't tell you whether a model's probability outputs are reliable for downstream decision thresholds. I implemented calibration analysis (Brier Score, ECE, reliability curves) as a standard evaluation step and built it into the reusable `ds_tools` package.

11. **GenAI technical leadership** — Analytics teams across LATAM lacked clear standards for LLM deployment and observability. I act as subject matter expert for Generative AI at Mercado Livre, defining architectural standards for LLM observability and safe deployment, and mentoring peers in Spanish.

12. **Quantitative foundation** — B.Sc. in Economics (UFV, quantitative focus: econometrics, statistical modelling) plus graduate coursework in Statistical Inference and Mathematical Modeling at UNICAMP. This background enables rigorous hypothesis testing and causal reasoning in feature engineering and model evaluation.
