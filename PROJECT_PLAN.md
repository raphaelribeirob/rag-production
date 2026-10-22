# Project Plan: rag-production

## Domain
Developer tooling and documentation Q&A: grounded question answering over technical docs.

## Target Companies
Pinecone, Writer, GitLab, Stone (BR), Wellhub (BR). These are companies whose AI roles
mention retrieval, evals or production LLM services.

## Evidence
Job descriptions (URLs checked live on 2026-10-22, HTTP 200 with the matching job title):
- Pinecone, Applied AI Engineer (Full-Stack): https://jobs.ashbyhq.com/pinecone/edf2a2a5-64e2-45b8-9f2e-4f354af350b8
- WRITER, AI Engineer: https://jobs.ashbyhq.com/writer/40f9e2b4-3284-4a20-b45d-0f5e5dd9108d
- GitLab, Senior AI Engineer: https://job-boards.greenhouse.io/gitlab/jobs/8565469002
- Stone, [GenAI] Senior AI Engineer (Florianopolis, hybrid): https://job-boards.greenhouse.io/stone/jobs/8017007003

Source of the list: `/workspace/career/ai-engineer-pivot-2026-10-22.md`.
TODO: quote the specific RAG, eval and production requirements from each posting.

Product docs / public data:
- pgvector: https://github.com/pgvector/pgvector
- Corpus (decided 2026-10-22): FastAPI docs, MIT License (c) 2018 Sebastián Ramírez, https://github.com/fastapi/fastapi

## Problem
- **User:** an engineer who maintains a docs assistant.
- **Input:** a natural-language question plus an ingested doc corpus.
- **Output:** a cited answer, latency, cost and a request id.
- **Failure modes:** wrong chunk retrieved; answer not supported by the citations; question has
  no answer in the corpus but the system answers anyway; LLM unavailable (falls back to extractive);
  ingest duplicates.
- **Success check:** CI eval gate hit@5 >= 0.80 and recall@5 >= 0.70 on the labeled set, and faithfulness
  >= 0.85 (local NLI).

## Selected Project
Production-style RAG API as specified in SPEC.md (AC1 to AC26). Due 2026-11-30.

# Build Plan

## First Version
Ingest the chosen corpus, serve `/query` with the extractive fallback, run `rag eval` in CI with
the gate. It will **not** include a UI, multi-tenancy, paid APIs or Langfuse (that is M6).

## Technologies to Demonstrate
- LLM: Ollama local (qwen2.5:1.5b / llama3.2:3b), extractive fallback
- RAG: sentence-transformers all-MiniLM-L6-v2, pgvector HNSW
- Agents/tools: none (non-goal)
- Evaluation: hit@5, recall@5, faithfulness, CI gate
- Tests: pytest, httpx, contract tests
- Monitoring/logging: structlog, OpenTelemetry, query_log
- Deployment/demo: docker-compose (local). Free deploy path TBD, no deploy yet.

## Open Decisions
- Decided: corpus = FastAPI docs (MIT). Faithfulness = local NLI gate >= 0.85, with the LLM judge offline only.
- Open: free deploy path.

## Progress log
- 2026-10-22: Postgres + pgvector store (psycopg 3), migrate script, docker-compose and CI
  service container (`pgvector/pgvector:pg16`), Neon deploy notes in README. Next: embeddings +
  ingest into the store, then `/query`.
