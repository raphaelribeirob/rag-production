# SECURITY.md: rag-production

## Scope
- Date: 2026-10-22 (America/Sao_Paulo). Mode: repository baseline of the local checkout plus the
  changes in this commit. Single local repository with no remote. Git history is 2-3 commits.
- Authorization assumed: inspection of this local repo only. No deployed system exists.
- Assets (planned): ingested corpus, `query_log` table, optional `API_KEY`/`HOSTED_LLM_API_KEY`. Trust boundaries (planned): HTTP clients to FastAPI; documents to the prompt (indirect prompt injection); app to Postgres/Ollama.

## Method
Repository-mode baseline review adapted from vibe-coding-toolkit `docs/prompts/13-security-review.md`
(Matheus Gomes, MIT, commit 790a895). The review was done by manual code tracing and focused
searches. No dedicated security skills or analyzers were installed or used. Note: the prompt
recommends a private report outside the repository. The owner explicitly asked for `SECURITY.md`
in the repo, so this file contains redacted content only.

## Coverage
- Reviewed: all tracked files (`src/`, `tests/`, `evals/`, `data/`, `.github/`, `pyproject.toml`,
  `.env.example`, `.gitignore`).
- Searches: hard-coded secrets (none found; `.env.example` holds empty or placeholder values only),
  shell/eval/deserialization sinks (none in `src/`), and CI triggers and permissions.
- Not run: dependency audit (the project declares no runtime dependencies yet), SAST tools,
  dynamic tests. There is no deployed target.

## Findings

| ID | Title | Severity | Status | Fix |
|---|---|---|---|---|
| R-1 | CI workflow had no explicit `permissions:` block, so `GITHUB_TOKEN` scope depended on repo/org defaults | low (hardening) | confirmed (static) | **fixed**: `permissions: contents: read`, `persist-credentials: false` |
| R-2 | Third-party actions pinned by tag (`@v4`, `@v5`) rather than commit SHA | low (hardening) | confirmed (static) | open: pin to SHAs before the repo goes public |
| R-3 | Planned API: auth is optional (`API_KEY` may be empty), so `/ingest` would be unauthenticated by default | medium if exposed (design) | hypothesis, code not built | proposed: require `API_KEY` for `/ingest` whenever it binds to non-localhost; add a test |
| R-4 | Planned retrieval feeds third-party docs into the LLM prompt (indirect prompt injection) | low (trusted MIT corpus, no tools) | hypothesis | proposed: keep generation tool-free and cite-only; add an AC26-style adversarial eval |

No critical, high or medium findings were substantiated in the inspected scope. That does not mean
the project is secure. Most of the attack surface is not built yet.

## Residual risk and next validation
- Re-review when `api.py`, ingestion and the SQL layer exist: parameterized queries, path traversal
  on `rag ingest <path>`, request size limits (SPEC caps the question at 2000 chars), log redaction.
- Run `pip-audit` once dependencies are declared (that needs authorization to send the manifest).
