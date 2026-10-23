# rag-production

![Status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

> **Status: work in progress.** Retrieval, retrieval eval and the NLI faithfulness gate run in CI.
> Providers: `extractive` (default), deterministic `mock` (used by the CI eval), local `ollama` and
> in-process `llamacpp` (both optional, never run in CI). CI faithfulness scores the **mock's** answers.
> A real model (Qwen2.5-7B-Instruct Q4_K_M via llama-cpp, local CPU) scored **0.647 faithfulness,
> below the 0.85 gate** (measured once on 2026-10-22, not in CI).
> See [evals/README.md](evals/README.md#real-llm-faithfulness-local-2026-10-22).

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
`/query` returns `answer`, `answer_source` (`extractive` = best chunk verbatim, or `generated`),
`abstained`, `citations` (`chunk_id`, `doc_id`, `score`), `latency_ms` (`embed`, `retrieve`,
`generate` when a provider runs, `total`), `cost_usd` (always 0.0: local only) and `provider`.
Local generation (optional, $0, no account), either:
- `pip install -e ".[local-llm]"`, download a GGUF from the official `Qwen/Qwen2.5-7B-Instruct-GGUF`
  repo (Q4_K_M; check its sha256), then `LLM_PROVIDER=llamacpp LLAMACPP_MODEL_PATH=/path/to/file.gguf`.
  The model runs in-process; nothing leaves the machine.
- or Ollama: `ollama pull qwen2.5:7b`, then `LLM_PROVIDER=ollama` (only `http://localhost` URLs).
TODO: terminal recording.

## Architecture
Ingest (load, heading-aware chunking, sentence-transformers embeddings) -> Postgres/pgvector (HNSW,
cosine) -> FastAPI `/query` (embed, retrieve top-k, build prompt with citations) -> LLM provider
chosen by `LLM_PROVIDER` behind a Protocol: `extractive` (no model), `mock` (deterministic, CI),
`ollama` (localhost) or `llamacpp` (in-process GGUF). `hosted_free` is reserved and raises `NotImplementedError`. Telemetry: structlog JSON, OpenTelemetry spans, `query_log` table.
Diagram: see SPEC.md. Built so far: chunking, embeddings, pgvector store, `/ingest`, `/query`
(retrieval + extractive or generated answer), providers. Not built: prompt-injection defenses,
telemetry, `query_log`.

## Tests
`pip install -e ".[dev]" && pytest -q` (needs the local Postgres above). Covers chunking,
embeddings, store, ingest idempotency, API contract (auth, input limits), providers (mock,
Ollama via a fake transport, llama-cpp via a fake model; no network, no model download) and the eval runner.

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
| faithfulness, CI: `mock` generated answers (47) | 1.000 (negative control 0.085) | >= 0.85 |
| faithfulness, extractive answer (45; CI before providers, run 39877673255) | 0.993 (negative control 0.055) | same gate |
| faithfulness, real LLM: Qwen2.5-7B-Instruct Q4_K_M, llama-cpp, local CPU (46) | **0.647** (negative control 0.150); generation p50 15.5 s, p95 29.7 s per query | fails 0.85; local only, not in CI |

Caveats: questions are hand-written (35 single-page, 12 multi-page, 10 unanswerable) and judged at
document level against a 40-page corpus. Retrieval questions are still fairly easy; treat the
numbers as a regression baseline. Abstention is a fixed top-1 cosine threshold (0.40), which is
crude and misses near-domain questions. Faithfulness is measured on the **extractive** answer (the
top chunk verbatim), so a high score is expected and only proves the judge works; it becomes
meaningful once a real model is measured. The real model scores 0.647. Some of that is wrong
claims and some is hedging or paraphrase the small judge does not credit (details in evals/README).
The gate, prompt and judge were not changed after seeing it. The mock copies retrieved sentences, so its 1.000 only
shows the generation path and judge are wired correctly; it says nothing about LLM quality. See [evals/README.md](evals/README.md).

## Limitations
- Corpus is 40 FastAPI tutorial pages (MIT); the eval set is small and easy (see Evals).
- Single tenant, no UI. `/ingest` requires `API_KEY` and fails closed when it is unset. Other endpoints have no auth in v1.
- Small local models: the extractive fallback gives safe but plain answers. A local 7B model is
  slow on CPU (about 15 s per answer) and only 0.647 faithful by the NLI judge.
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
