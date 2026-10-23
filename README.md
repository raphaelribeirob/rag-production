# rag-production

![Status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

> **Status: work in progress.** Retrieval pipeline, retrieval eval and NLI faithfulness gate run in
> CI. Generation (LLM providers) is not built yet, so faithfulness currently scores the extractive answer.

A production-style RAG API: FastAPI, Postgres + pgvector, local embeddings and
quality evals gated in CI. Full requirements: [SPEC.md](SPEC.md) (AC1 to AC26).

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
```bash
docker compose up -d postgres && python tools/fetch_corpus.py && python tools/migrate.py
API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))") \
  uvicorn --factory rag_production.api:build_default_app   # API_KEY stays in this shell only
curl -X POST localhost:8000/ingest -H "X-API-Key: $API_KEY"   # same shell/env
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "How do I upload a file?", "top_k": 5}'
```
`/query` returns `answer` (best chunk verbatim; extractive, no LLM yet), `citations`
(`chunk_id`, `doc_id`, `score`), `latency_ms` (`embed`, `retrieve`, `total`), `cost_usd` (0.0:
local model) and `provider`. TODO: terminal recording.

## Architecture
Ingest (load, heading-aware chunking, sentence-transformers embeddings) -> Postgres/pgvector (HNSW,
cosine) -> FastAPI `/query` (embed, retrieve top-k, build prompt with citations) -> LLM provider
chosen by `LLM_PROVIDER`: `ollama` (local), `hosted_free` (optional) or `extractive`
(no-model fallback, used in CI). Telemetry: structlog JSON, OpenTelemetry spans, `query_log` table.
Diagram: see SPEC.md. Built so far: chunking, embeddings, pgvector store, `/ingest`, `/query`
(retrieval + extractive answer). Not built: LLM providers, telemetry, `query_log`.

## Tests
`pip install -e ".[dev]" && pytest -q` (needs the local Postgres above). Covers chunking,
embeddings, store, ingest idempotency, API contract (auth, input limits) and the eval runner.

## Evals
Hit rate@5, recall@5 and faithfulness on a hand-labeled set built from the FastAPI docs.
CI gate: hit@5 >= 0.80, recall@5 >= 0.70 and faithfulness >= 0.85. Faithfulness is scored by a
local NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`), which is free and deterministic.
An LLM judge runs only offline, and its score is reported here without gating.
Retrieval numbers below are measured (`python -m rag_production.eval_runner`, full set, run in CI
on every push).

| Metric (answerable, n=47) | Measured (2026-10-22) | Gate |
|---|---|---|
| hit@5 (doc level) | 1.00 (single 1.00, multi 1.00) | >= 0.80 |
| recall@5 (doc level) | 0.961 (single 0.986, multi 0.889) | >= 0.70 (stretch 0.80) |
| abstention on 10 unanswerable | 70% (3 near-domain answered wrongly) | reported, not gated |
| false abstention on answerable | 0% | reported, not gated |
| faithfulness (45 non-abstained answerable, extractive answer) | 0.993 (negative control 0.055) | >= 0.85 |

Caveats: questions are hand-written (35 single-page, 12 multi-page, 10 unanswerable) and judged at
document level against a 40-page corpus. Retrieval questions are still fairly easy; treat the
numbers as a regression baseline. Abstention is a fixed top-1 cosine threshold (0.40), which is
crude and misses near-domain questions. Faithfulness is measured on the **extractive** answer (the
top chunk verbatim), so a high score is expected and only proves the judge works; it becomes
meaningful once generated answers exist. See [evals/README.md](evals/README.md).

## Limitations
- Corpus is 40 FastAPI tutorial pages (MIT); the eval set is small and easy (see Evals).
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
