# rag-production

![Status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

> **Status: work in progress.** All results and metrics are TODO; no numbers have been measured yet.

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
Hit rate@5, recall@5 and faithfulness on a hand-labeled set built from the FastAPI docs.
CI gate: hit@5 >= 0.80, recall@5 >= 0.70 and faithfulness >= 0.85. Faithfulness is scored by a
local NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`), which is free and deterministic.
An LLM judge runs only offline, and its score is reported here without gating.
**TODO: no eval has run yet. Both result tables are empty.**

| Metric | CI (NLI) | Offline LLM judge |
|---|---|---|
| hit@5 | TODO | n/a |
| recall@5 | TODO | n/a |
| faithfulness | TODO | TODO | See [evals/README.md](evals/README.md).

## Limitations
- The corpus (FastAPI docs, MIT) has not been ingested yet, so there are no results.
- Single tenant, no UI. `/ingest` requires `API_KEY` and fails closed when it is unset. Other endpoints have no auth in v1.
- Small local models: the extractive fallback gives safe but plain answers.
- Langfuse dashboards are a separate deliverable (due 2026-12-15).

## Attribution
The corpus is the FastAPI documentation, MIT License, Copyright (c) 2018 Sebastián Ramírez,
https://github.com/fastapi/fastapi.

## Key files
`SPEC.md`, `PROJECT_PLAN.md`, `src/rag_production/`, `evals/`, `.github/workflows/ci.yml`.

## Agent docs
`AGENTS.md` (agent rules), `SECURITY.md` (security review), `HARNESS.md` (agent-harness evaluation). Gates and templates are adapted from vibe-coding-toolkit (Matheus Gomes, MIT, commit 790a895).

## Local Postgres + pgvector

```bash
docker compose up -d postgres          # pgvector/pgvector:pg16
cp .env.example .env                   # DATABASE_URL points at the local container
python tools/migrate.py                # CREATE EXTENSION vector, chunks table, HNSW cosine index
pytest -q                              # store tests run against the real database
```

## Deploy on Neon (free tier)

Not performed by this repo; steps for a reviewer who wants a hosted database at $0:

1. Create a free account and a new project at <https://neon.tech> (Postgres 16).
2. In the Neon SQL editor run `CREATE EXTENSION IF NOT EXISTS vector;`
   (`tools/migrate.py` also runs it, but enabling it once up front surfaces permission issues early).
3. Copy the connection string from the dashboard and set it as `DATABASE_URL` in your
   deployment's secret store (never commit it), keeping `sslmode=require`:
   `postgresql://USER:PASSWORD@HOST.neon.tech/DBNAME?sslmode=require`
4. Run `python tools/migrate.py` once against that URL to create the table and HNSW index.
