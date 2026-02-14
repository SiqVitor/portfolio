# Vitor Rodrigues — ML Engineering Portfolio

**Senior Machine Learning Engineer** — fraud prevention, GenAI agents, production ML systems.

## TL;DR

- **Fraud prevention at scale**: Designed and deployed end-to-end ML pipelines (LightGBM, PyTorch with entity embeddings) processing 200M+ rows/month at Mercado Livre. Savings of $30k–$50k USD/month in prevented chargebacks and account takeovers.
- **GenAI agent automation**: Built an autonomous agent framework (LangChain/LangGraph) to automate ATO report review, reducing manual workload by 70% and freeing 15–20 analysts.
- **Production ML ownership**: Full lifecycle — BigQuery/Databricks ingestion, MLflow experiment tracking and model registry, Docker containerised deployment (GCP Vertex AI), Prometheus/Grafana monitoring, canary rollout with automated rollback.

## Tech Stack

| Area | Tools |
|------|-------|
| ML / DL | scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch (DDP, AMP, TorchScript, ONNX) |
| GenAI | LangChain, LangGraph, RAG (ChromaDB), Groq/Llama 3 |
| Data | Python, SQL, BigQuery, Azure Databricks, pandas, NumPy |
| MLOps | MLflow (tracking + registry), Docker, GCP Vertex AI, CI/CD |
| Monitoring | Prometheus, Grafana, Evidently AI, PSI/KS drift detection |
| Serving | FastAPI, containerised endpoints, canary rollout |

## Case Studies

| Project | Summary | Demo |
|---------|---------|------|
| [Fraud Detection](fraud_detection/) | End-to-end fraud pipeline: IEEE-CIS + synthetic data, LightGBM, PyTorch, calibration, monitoring | `bash fraud_detection/demo/run_demo.sh` |
| [ARGUS — GenAI Agent](genai_agent/) | Multi-agent RAG system with citation evaluation (WIP) | `bash genai_agent/demo/run_local_eval.sh` |
| [Real-Time ML System](realtime_ml_system/) | Online inference pipeline: streaming features, latency tracking, batch/online separation | `bash realtime_ml_system/demo/run_demo.sh` |
| [ML Platform](ml_platform/) | ML lifecycle orchestration: validation → training → evaluation → model registry | `bash ml_platform/demo/run_pipeline.sh` |

### Supporting Projects

| Project | Description |
|---------|-------------|
| [ds_tools](ds_tools/) | Reusable ML toolkit — sklearn transformers, evaluation reports, drift monitoring |
| [Kaggle Competitions](kaggle/) | House Prices (top 12.5%), Titanic, applied statistics |

## Run a 10-Minute Demo

```bash
# Option 1: Docker (recommended — nothing to install)
docker build -t portfolio-demo .
docker run --rm portfolio-demo

# Option 2: Local Python (requires Python 3.10+)
pip install -e ds_tools/ && pip install lightgbm matplotlib
bash fraud_detection/demo/run_demo.sh
bash realtime_ml_system/demo/run_demo.sh
bash ml_platform/demo/run_pipeline.sh
```

**Expected outputs:**
- `fraud_detection/demo/results/summary.json` — ROC-AUC, Brier Score, ECE, classification report
- `realtime_ml_system/demo/results/summary.json` — latency p50/p95/p99, throughput, online AUC
- `ml_platform/demo/results/metrics.json` — champion model, validation report, registered version

## Documentation

| Document | Audience |
|----------|----------|
| [architecture.md](architecture.md) | Technical interviewers — cross-project design patterns |
| [model_card_template.md](model_card_template.md) | ML governance — fillable model card |

---

Built by **Vitor de Siqueira Rodrigues** · [LinkedIn](https://linkedin.com/in/r-vitor) · [GitHub](https://github.com/SiqVitor)
