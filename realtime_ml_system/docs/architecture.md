# Architecture Decisions — Real-Time ML System

Trade-off analysis for an online inference pipeline serving fraud scores at transaction time.

## Online vs. Batch Separation

| Concern | Online path | Batch path |
|---------|-------------|------------|
| **Purpose** | Score individual events at request time | Train models on historical labelled data |
| **Latency** | < 50ms end-to-end | Hours (acceptable) |
| **Data** | Single event + pre-computed features | Full dataset (millions of rows) |
| **Model update** | Hot-reload from model registry | Scheduled retraining (weekly) |
| **Failure impact** | Transactions scored with fallback (default approve at low risk) | Retraining delayed — stale model serves until fixed |

**Why separate**: Training requires heavy compute (GPU for PyTorch, full dataset traversal). Serving requires low latency and high availability. Coupling them creates fragile systems where a training failure blocks serving.

## Feature Store Design

### Feature Freshness Tiers

| Tier | Refresh | Source | Examples |
|------|---------|--------|----------|
| **Real-time** | Per-event | Event payload | `transaction_amount`, `device_type`, `ip_country` |
| **Near-real-time** | 1–5 min | Streaming aggregation | `txn_count_last_hour`, `distinct_merchants_30min` |
| **Batch** | 6h | Scheduled pipeline | `avg_daily_spend_30d`, `merchant_freq_encoding`, `velocity_7d` |

### Lookup Strategy

At inference time, the feature assembler:
1. Extracts real-time features from the event payload (zero latency cost)
2. Looks up pre-computed features from the feature store (Redis in production, in-memory dict in demo) — target < 2ms
3. Merges into a single feature vector and passes to the model

**Trade-off**: Near-real-time features (tier 2) add 1–5 min staleness but capture short-term velocity patterns that batch features miss. The cost is maintaining a streaming aggregation pipeline (e.g., Flink or Kafka Streams). In the demo, these are simulated as pre-computed values.

## Latency Budget

| Component | Budget | Notes |
|-----------|--------|-------|
| Event parsing | < 0.5ms | JSON deserialization |
| Feature store lookup | < 2ms | Redis GET (single key) |
| Model inference | < 5ms | LightGBM predict (single row) |
| Decision + logging | < 1ms | Threshold comparison + async log |
| **Total** | **< 10ms p50** | Network overhead adds ~5–20ms in production |

**Why LightGBM over PyTorch for online serving**: LightGBM single-row inference is ~0.1ms (CPU). PyTorch entity-embedding model is ~5–10ms (CPU) or ~1ms (GPU). For the online path where p95 < 50ms including network, LightGBM provides sufficient margin. PyTorch is used in the batch re-scoring path where latency is not constrained.

## Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|------------|
| **Feature store unavailable** | Missing pre-computed features | Serve with real-time features only (degraded accuracy, not outage). Log degraded predictions separately. |
| **Model file corrupted** | No scoring possible | Keep previous model version in memory. Atomic model swap: load new → verify → swap pointer. |
| **Event stream backpressure** | Scoring delayed | Horizontal scaling (add consumer instances). If delay > SLO, route to fast-path (rules-based fallback). |
| **Model drift** | Increased false positives/negatives | Daily PSI monitoring on incoming features. Weekly batch eval on labelled cohort. Alert → retrain. |
| **Downstream service timeout** | Decision not applied | Async decision propagation with retry queue. Default to "approve" for low-risk transactions (amount < threshold). |

## Scaling Considerations

- **Horizontal**: Stateless inference workers behind a load balancer. Each worker holds a copy of the model in memory. Adding workers is linear throughput scaling.
- **Feature store**: Redis cluster with read replicas. Feature computation is a separate batch pipeline; reads are cheap.
- **Bottleneck**: Feature store writes during batch refresh. Mitigated by double-buffering (write to shadow table, swap atomically).
- **Cost**: LightGBM on CPU is ~$0.001 per 1000 predictions. PyTorch on GPU is ~$0.01 per 1000 predictions. Online path uses LightGBM; batch path can use either.
