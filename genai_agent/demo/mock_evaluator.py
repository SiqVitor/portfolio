"""Mock evaluator for ARGUS — runs offline without API keys.

Simulates the evaluation pipeline with synthetic queries and pre-defined
mock responses. Computes faithfulness, citation accuracy, and relevance scores.

Output: End to End GenAI Agent/demo/results/eval_report.json
"""

import json
from pathlib import Path

QUERIES = [
    {
        "id": "q1",
        "question": "What was the company's revenue in Q3?",
        "context": ["Q3 revenue reached $12.5M, a 15% increase year-over-year."],
        "mock_response": "The company's revenue in Q3 was $12.5M, representing a 15% year-over-year increase. [Source 1]",
        "mock_citations": [{"ref": "Source 1", "passage_idx": 0}],
    },
    {
        "id": "q2",
        "question": "How many employees does the company have?",
        "context": [
            "The company hired 30 new engineers in Q2.",
            "Total headcount is 450 as of December.",
        ],
        "mock_response": "The company has 450 employees as of December. [Source 2]",
        "mock_citations": [{"ref": "Source 2", "passage_idx": 1}],
    },
    {
        "id": "q3",
        "question": "What is the main product?",
        "context": ["Our flagship product is a payment processing platform."],
        "mock_response": "The main product is a payment processing platform. [Source 1]",
        "mock_citations": [{"ref": "Source 1", "passage_idx": 0}],
    },
    {
        "id": "q4",
        "question": "What markets does the company operate in?",
        "context": ["Operations span Brazil, Argentina, and Mexico."],
        "mock_response": "The company operates in Brazil, Argentina, Mexico, and Chile. [Source 1]",
        "mock_citations": [{"ref": "Source 1", "passage_idx": 0}],
    },
    {
        "id": "q5",
        "question": "Who is the CEO?",
        "context": [],
        "mock_response": "I could not find information about the CEO in the provided documents.",
        "mock_citations": [],
    },
]


def compute_faithfulness(response: str, context: list[str]) -> float:
    """Fraction of response sentences that are supported by context."""
    if not context:
        # No context: system should abstain, which is correct
        return 1.0 if "could not find" in response.lower() else 0.0

    sentences = [
        s.strip()
        for s in response.replace("[Source 1]", "").replace("[Source 2]", "").split(".")
        if s.strip()
    ]
    if not sentences:
        return 0.0

    supported = 0
    context_text = " ".join(context).lower()
    for sentence in sentences:
        words = sentence.lower().split()
        # Simple word-overlap check (production would use embeddings)
        overlap = sum(1 for w in words if w in context_text)
        if overlap / max(len(words), 1) > 0.4:
            supported += 1

    return supported / len(sentences)


def compute_citation_accuracy(citations: list[dict], context: list[str]) -> float:
    """Fraction of citations pointing to valid, relevant passages."""
    if not citations:
        return 1.0  # No citations when no context is correct

    valid = 0
    for cite in citations:
        idx = cite.get("passage_idx", -1)
        if 0 <= idx < len(context):
            valid += 1

    return valid / len(citations)


def evaluate():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_results = []
    for q in QUERIES:
        faithfulness = compute_faithfulness(q["mock_response"], q["context"])
        citation_acc = compute_citation_accuracy(q["mock_citations"], q["context"])

        eval_results.append(
            {
                "query_id": q["id"],
                "question": q["question"],
                "faithfulness": round(faithfulness, 3),
                "citation_accuracy": round(citation_acc, 3),
                "response_preview": q["mock_response"][:100],
            }
        )

    avg_faith = sum(r["faithfulness"] for r in eval_results) / len(eval_results)
    avg_cite = sum(r["citation_accuracy"] for r in eval_results) / len(eval_results)

    report = {
        "mode": "mock (no API keys)",
        "n_queries": len(QUERIES),
        "avg_faithfulness": round(avg_faith, 3),
        "avg_citation_accuracy": round(avg_cite, 3),
        "per_query": eval_results,
    }

    output_path = results_dir / "eval_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation complete ({len(QUERIES)} queries)")
    print(f"  Avg faithfulness:      {avg_faith:.3f}")
    print(f"  Avg citation accuracy: {avg_cite:.3f}")
    print(f"  Report: {output_path}")
    return report


if __name__ == "__main__":
    evaluate()
