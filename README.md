# rag-production

A production-style RAG API: FastAPI, Postgres + pgvector, local embeddings and
quality evals gated in CI. Full requirements: [SPEC.md](SPEC.md) (AC1 to AC26).

> Status: **skeleton**. Sections marked TODO are not built yet. No results are claimed.

## Problem
Teams that put RAG in front of users rarely know whether retrieval is good enough, or whether
a change made it worse. This project treats retrieval quality, latency and cost as measured,
CI-checked properties instead of demo impressions.

## User
- An engineer running a docs Q&A service who needs grounded answers and a regression signal.
- A hiring reviewer who wants to run it locally in under 10 minutes at $0.

## Input / Output
- **Input:** a question (1 to 2000 chars) and an optional `top_k` (default 5), via `POST /query`.
  Documents come in through `rag ingest <path>` / `POST /ingest`.
- **Output:** an answer, citations (chunk id, doc id, source, score), a per-stage latency
  breakdown, estimated cost, the provider used and a request id. See the SPEC API contract.

## Demo
TODO: sample request and response, a terminal recording and the `docker compose up` walkthrough.

## Architecture
Ingest (load, heading-aware chunking, sentence-transformers embeddings) -> Postgres/pgvector (HNSW,
cosine) -> FastAPI `/query` (embed, retrieve top-k, build prompt with citations) -> LLM provider
chosen by `LLM_PROVIDER`: `ollama` (local), `hosted_free` (optional) or `extractive`
(no-model fallback, used in CI). Telemetry: structlog JSON, OpenTelemetry spans, `query_log` table.
Diagram: see SPEC.md. TODO: `docker-compose.yml`.

## Tests
`pip install -e ".[dev]" && pytest -q`. Right now there are only smoke tests for config and metric
functions. TODO: API contract tests, ingestion idempotency tests and DB integration tests.

## Evals
Hit rate@5, recall@5 and faithfulness on a hand-labeled set. CI gate: hit@5 >= 0.80 and
recall@5 >= 0.70. **TODO: no eval has run yet.** See [evals/README.md](evals/README.md).

## Limitations
- Corpus not chosen yet (TBD). There is no data, so there are no results.
- Single tenant, API-key auth at most, no UI.
- Small local models: the extractive fallback gives safe but plain answers.
- Langfuse dashboards are a separate deliverable (due 2026-12-15).

## Key files
`SPEC.md`, `PROJECT_PLAN.md`, `src/rag_production/`, `evals/`, `.github/workflows/ci.yml`.
