# Evaluation Methodology — ARGUS

How we measure whether ARGUS responses are faithful, relevant, and properly cited.

## Metrics

### 1. Faithfulness Score (0.0–1.0)

**Definition**: Fraction of claims in the generated response that are supported by the retrieved context passages.

**Scoring**:
- Extract individual factual claims from the response (sentence-level decomposition)
- For each claim, check if any retrieved passage contains supporting evidence
- `faithfulness = supported_claims / total_claims`

**Example**:
```
Response: "Revenue grew 15% in Q3. The team expanded to 50 people."
Retrieved context: "Q3 revenue increased by 15% year-over-year."

Claim 1: "Revenue grew 15% in Q3" → supported (passage matches) → 1
Claim 2: "The team expanded to 50 people" → not supported → 0

Faithfulness = 1/2 = 0.50
```

**Target**: ≥ 0.85 on benchmark queries.

### 2. Citation Accuracy (0.0–1.0)

**Definition**: Fraction of citations in the response that map to an actual source passage containing relevant information.

**What counts as a citation**: Any explicit reference to a source document or passage index (e.g., `[Source 1]`, `[doc_3, p.12]`).

**Scoring**:
- Extract all citations from the response
- For each citation, verify that the referenced source exists in the retrieval set
- Verify that the referenced source actually supports the claim it's attached to
- `citation_accuracy = valid_citations / total_citations`

**Target**: ≥ 0.90 on benchmark queries.

### 3. Answer Relevance (0.0–1.0)

**Definition**: Does the response actually address the question asked?

**Scoring**: Semantic similarity between the question and the response, computed via embedding cosine similarity. This catches cases where the model generates faithful but off-topic responses.

**Target**: ≥ 0.80 on benchmark queries.

### 4. Latency

**Definition**: Wall-clock time from query submission to full response delivery.

| Metric | Target |
|--------|--------|
| p50 | < 3s |
| p95 | < 8s |
| p99 | < 15s |

## Evaluation Pipeline

```
Synthetic Queries (queries.json)
        │
        ▼
   Mock LLM / Real LLM
        │
        ▼
   Response Generation
        │
        ▼
   Metric Computation
   (faithfulness, citation accuracy, relevance)
        │
        ▼
   eval_report.json
```

### Mock Mode (default — no API keys)

Uses a deterministic mock that returns pre-defined responses with citations. Useful for validating the evaluation harness itself and for CI.

### Live Mode (requires API keys)

Set `GROQ_API_KEY` and `GOOGLE_API_KEY` in `.env` to run against real LLM endpoints.

## Benchmark Dataset

The evaluation uses 5 synthetic queries with known-correct answers and expected citations. This is intentionally small — the purpose is to validate the evaluation *pipeline*, not to benchmark model quality (which requires domain-specific datasets).

## Limitations

- Mock mode tests the evaluation harness, not the actual LLM quality
- Faithfulness scoring depends on claim extraction quality
- Citation accuracy assumes structured citation format — unstructured references are not captured
- Answer relevance via embedding similarity is a proxy; manual review is needed for edge cases
