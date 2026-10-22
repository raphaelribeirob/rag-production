# SECURITY.md: rag-production

Status: work in progress. Updated 2026-10-22 (America/Sao_Paulo). Only controls with a test are
listed as covered; everything else is in "Not covered yet". No deployed system exists.

## Scope and method
- Assets: ingested corpus (FastAPI docs, MIT), Postgres/pgvector store, `API_KEY` (ingest),
  `DATABASE_URL` (may embed a password, e.g. Neon), optional `HOSTED_LLM_API_KEY`.
- Trust boundaries: HTTP clients -> FastAPI (planned); documents -> prompt (planned); app -> Postgres.
- Method: manual code review adapted from vibe-coding-toolkit `docs/prompts/13-security-review.md`
  (Matheus Gomes, MIT, commit 790a895). No SAST/DAST or dependency audit run yet.

## Controls mapped to OWASP (each backed by a test)

LLM = OWASP Top 10 for LLM Applications 2025; A = OWASP Top 10 (2021) for the HTTP API.

| Control | OWASP | Proving test |
|---|---|---|
| Ingest fails closed: 503 when `API_KEY` unset | A01 Broken Access Control; LLM04 Data and Model Poisoning (no anonymous corpus writes) | `tests/test_ingest_auth.py::test_ingest_fails_closed_when_key_unset` |
| Ingest rejects missing/empty/wrong key with 401 (constant-time compare) | A01 Broken Access Control; A07 Identification and Authentication Failures | `tests/test_ingest_auth.py::test_ingest_rejects_missing_or_wrong_key` |
| Correct key is accepted (control is not deny-all) | A01 | `tests/test_ingest_auth.py::test_ingest_accepts_matching_key` |
| `Settings` repr hides `API_KEY` and `DATABASE_URL` (no secrets in logs/tracebacks) | LLM02 Sensitive Information Disclosure; A09 Security Logging and Monitoring Failures | `tests/test_ingest_auth.py::test_settings_repr_does_not_leak_api_key` |
| SQL values are parameterized; injection payload stored as data | A03 Injection | `tests/test_store.py::test_sql_injection_payload_is_stored_as_data` |
| Table identifier allow-listed before any SQL is built | A03 Injection | `tests/test_store.py::test_invalid_table_name_rejected` |
| Embedding dimension enforced on write | LLM08 Vector and Embedding Weaknesses (partial: integrity of stored vectors only) | `tests/test_store.py::test_wrong_dimension_rejected` |
| Only free/local LLM providers accepted (no paid API by misconfiguration) | A05 Security Misconfiguration; LLM10 Unbounded Consumption (partial: cost only) | `tests/test_smoke.py::test_invalid_provider_rejected` |
| Safe default provider is `extractive` (no model call) | A05 Security Misconfiguration | `tests/test_smoke.py::test_default_settings_use_no_model_fallback` |
| GitHub Actions pinned by full commit SHA | LLM03 Supply Chain; A08 Software and Data Integrity Failures | `tests/test_repo_hardening.py::test_actions_pinned_by_full_sha` |
| CI token read-only, `persist-credentials: false` | A08; A05 | `tests/test_repo_hardening.py::test_workflow_token_is_read_only` |
| `.env.example` holds placeholders only | LLM02 Sensitive Information Disclosure; A02 Cryptographic Failures (secret handling) | `tests/test_repo_hardening.py::test_env_example_has_only_placeholder_secrets` |
| `.env` is gitignored | LLM02 | `tests/test_repo_hardening.py::test_dotenv_is_gitignored` |

## Not covered yet (no test, so not claimed)
- LLM01 Prompt Injection (incl. indirect via retrieved docs): generation not built; planned
  adversarial eval (SPEC AC26-style) and tool-free, cite-only answers.
- LLM05 Improper Output Handling, LLM07 System Prompt Leakage, LLM09 Misinformation (the NLI
  faithfulness gate >= 0.85 is planned, not implemented), LLM06 Excessive Agency (design has no
  tools; nothing to test yet).
- LLM08 beyond dimension checks: access control per document/tenant, embedding inversion.
- LLM10 rate limiting, request size limits (SPEC: question <= 2000 chars), timeouts.
- A01/A07 on real HTTP endpoints: `api.py` is not wired; `authorize_ingest` is tested as a function.
- A02 TLS to the DB: Neon requires `sslmode=require` (documented in README), not enforced in code.
- A04 Insecure Design threat model, A06 Vulnerable Components (`pip-audit` not run), A10 SSRF
  (no outbound fetches yet; Ollama URL is config-controlled).

## Earlier findings
| ID | Title | Status |
|---|---|---|
| R-1 | CI had no explicit `permissions:` | fixed (`contents: read`), tested |
| R-2 | Actions pinned by tag | fixed (SHA pins), tested |
| R-3 | `/ingest` would be open if `API_KEY` empty | fixed in design (`authorize_ingest` fails closed), tested; endpoint wiring TODO |
| R-4 | Indirect prompt injection via corpus | open (hypothesis), see LLM01 above |
