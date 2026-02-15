# ARGUS — GenAI Agent Case Study (WIP)

Multi-agent system for automated research, data analysis, and document-grounded Q&A. Built as a demonstration of production GenAI engineering practices.

**Status**: Work in progress. Core agent orchestration and RAG pipeline are functional. Evaluation pipeline and production hardening are under active development.

## Objective

Build a multi-agent system that can:
1. Answer questions grounded in uploaded documents (RAG)
2. Perform web research with source citations
3. Execute Python code for data analysis and visualisation

Every response must include source citations. No hallucinated claims — if the system cannot find supporting evidence, it must say so.

## Architecture

```
User → Streamlit UI → FastAPI → LangGraph Orchestrator
                                    ├── Research Agent (web search + summarisation)
                                    ├── Data Analyst Agent (code execution + visualisation)
                                    └── RAG Agent (vector search + citations)
```

| Component | Technology |
|-----------|------------|
| LLM | Groq (Llama 3.3) |
| Orchestration | LangGraph (state machine routing) |
| RAG | LangChain + ChromaDB (hybrid: dense + BM25) |
| Embeddings | Google Gemini |
| Vector DB | ChromaDB |
| API | FastAPI (SSE streaming) |
| Frontend | Streamlit |
| Storage | PostgreSQL (conversation), Redis (cache) |

## Safe-by-Design

- **Sandboxed execution**: Code execution agent runs in an isolated environment
- **Grounded responses**: All RAG responses include source citations from retrieved documents
- **No autonomous external actions**: Tool outputs are presented to the user, never auto-executed
- **Citation tracking**: Every claim is mapped to a specific source passage

## What Is Delivered

- [x] LangGraph multi-agent orchestration with intent classification
- [x] Hybrid RAG pipeline (dense + BM25) with source citations
- [x] FastAPI backend with SSE streaming
- [x] Streamlit frontend with real-time updates
- [x] Docker Compose setup (API + DB + cache)

## What Is WIP

- [ ] Automated evaluation pipeline (faithfulness, citation accuracy)
- [ ] Performance benchmarks on standard Q&A datasets
- [ ] Production monitoring and observability
- [ ] Rate limiting and authentication

## Run Local Evaluation (Mock Mode)

```bash
bash genai_agent/demo/run_local_eval.sh
```

This runs a local, offline evaluation on synthetic queries using a mock LLM (no API keys required). To use a real LLM, copy `.env.example` to `.env` and add your API keys.

Output: `genai_agent/demo/results/eval_report.json`

## Related Files

- [Source code](../src/) — complete agent implementation
- [Evaluation methodology](evaluation.md) — citation and faithfulness metrics
