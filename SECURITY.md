# SECURITY.md: rag-production

Status: work in progress. Updated 2026-10-22 (America/Sao_Paulo). Only controls with a test are
listed as covered; everything else is in "Not covered yet". No deployed system exists.

## Scope and method
- Assets: ingested corpus (FastAPI docs, MIT), Postgres/pgvector store, `API_KEY` (ingest),
  `DATABASE_URL` (may embed a password, e.g. Neon), optional `HOSTED_LLM_API_KEY`.
- Trust boundaries: HTTP clients -> FastAPI; documents -> prompt (planned, no LLM yet); app -> Postgres.
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
| `POST /ingest` over HTTP: 401 without/with wrong `X-API-Key` | A01; A07 | `tests/test_api.py::test_ingest_requires_key` |
| `POST /ingest` over HTTP: 503 when no `API_KEY` configured | A01; LLM04 | `tests/test_api.py::test_ingest_disabled_without_configured_key` |
| `/ingest` reads only the server-configured corpus dir; client paths ignored (no path traversal) | A01; LLM04 | `tests/test_api.py::test_ingest_does_not_accept_client_supplied_paths` |
| `/query` input limits: question 1..2000 chars, `top_k` 1..20, required fields (422) | LLM10 Unbounded Consumption (partial); A04 Insecure Design | `tests/test_api.py::test_query_input_limits` |
| Per-query cost and latency returned (cost visibility; $0 local model) | LLM10 (partial: observability) | `tests/test_api.py::test_ingest_then_query_returns_citations_cost_latency` |
| Corpus fetched over HTTPS from a pinned 40-char upstream commit, fixed page allow-list | LLM03 Supply Chain; LLM04 Data and Model Poisoning | `tests/test_repo_hardening.py::test_corpus_source_pinned_to_full_commit_and_https` |
| `/query` abstains (empty answer, `abstained: true`) when top-1 similarity < `ABSTAIN_THRESHOLD` | LLM09 Misinformation (partial: 70% abstention on unanswerable eval set) | `tests/test_api.py::test_query_abstains_below_threshold` |
| CI eval gate: NLI faithfulness >= 0.85 (sentences entailed by retrieved chunks); a missing score fails the gate | LLM09 Misinformation (partial: CI scores the deterministic mock's answers, not a real LLM) | `tests/test_eval_runner.py::test_gate_failures_include_faithfulness_and_missing_metrics`, `tests/test_faithfulness.py::test_real_nli_judge_entailment_vs_contradiction` |
| Failure taxonomy of answers (supported / hallucination / evasive / paraphrase) with rules fixed before use; agreement with hand labels computed, not assumed | LLM09 (partial: offline measurement, not a runtime control; 0.65 agreement on 20 answers) | `tests/test_taxonomy.py::test_sentence_rules_in_order`, `tests/test_taxonomy.py::test_answer_label_priority_and_score`, `tests/test_taxonomy.py::test_agreement_with_hand_labels` |
| Dev/test question splits frozen by sha256, so prompt tuning cannot silently use test | LLM09 (eval integrity) | `tests/test_eval_splits.py::test_split_hashes_match_frozen_files` |
| Ollama provider only talks to `http://localhost` (no data sent to remote hosts, no spend) | LLM02; LLM10 (cost); A10 SSRF (partial: provider URL only) | `tests/test_providers.py::test_ollama_only_talks_to_localhost`, `tests/test_providers.py::test_ollama_settings_from_env_still_localhost_only` |
| Generation is deterministic and context-only by instruction (temperature 0, seed 0, "only the context" prompt) | LLM09 (partial: instruction, not enforcement) | `tests/test_providers.py::test_ollama_request_is_local_deterministic_and_context_only` |
| llama-cpp provider runs in-process from a local model file only (no network, no spend); fails closed if `LLAMACPP_MODEL_PATH` is empty | LLM02; LLM10 (cost); A05 | `tests/test_providers.py::test_llamacpp_selected_by_settings_and_requires_model_path` |
| llama-cpp generation is deterministic and context-only by instruction (temperature 0, seed 0) | LLM09 (partial: instruction, not enforcement) | `tests/test_providers.py::test_llamacpp_in_process_deterministic_context_only` |
| No hosted provider: `hosted_free` raises `NotImplementedError` | LLM10 (cost); A05 | `tests/test_providers.py::test_get_provider_by_setting` |
| GitHub Actions pinned by full commit SHA | LLM03 Supply Chain; A08 Software and Data Integrity Failures | `tests/test_repo_hardening.py::test_actions_pinned_by_full_sha` |
| CI token read-only, `persist-credentials: false` | A08; A05 | `tests/test_repo_hardening.py::test_workflow_token_is_read_only` |
| `.env.example` holds placeholders only | LLM02 Sensitive Information Disclosure; A02 Cryptographic Failures (secret handling) | `tests/test_repo_hardening.py::test_env_example_has_only_placeholder_secrets` |
| `.env` is gitignored | LLM02 | `tests/test_repo_hardening.py::test_dotenv_is_gitignored` |

## Not covered yet (no test, so not claimed)
- LLM01 Prompt Injection (incl. indirect via retrieved docs): the Ollama and llama-cpp providers put retrieved
  chunks into the prompt with no injection defense or test; planned adversarial eval (SPEC AC26).
- LLM05 Improper Output Handling, LLM07 System Prompt Leakage, LLM09 for real LLM output
  (measured locally, Qwen2.5-7B Q4_K_M via llama-cpp: faithfulness 0.647 baseline, 0.605 with
  prompt v5 after an env rebuild, all **below** the 0.85 gate; not gated in CI, which gates the
  mock only; no runtime check blocks an unsupported sentence), LLM06 Excessive Agency
  (design has no tools; nothing to test yet).
- LLM08 beyond dimension checks: access control per document/tenant, embedding inversion.
- LLM10 rate limiting, request body size limits at the server, timeouts.
- A07 brute force / rate limiting on `/ingest`; `/query` and `/health` are unauthenticated by design (v1).
- Embedding model integrity: weights downloaded from Hugging Face by name, not pinned by revision/hash.
- Local GGUF integrity: sha256 was checked by hand for the measured run (evals/README); the code
  does not verify the model file.
- A02 TLS to the DB: Neon requires `sslmode=require` (documented in README), not enforced in code.
- A04 Insecure Design threat model, A06 Vulnerable Components (`pip-audit` not run), A10 SSRF
  beyond the localhost-only Ollama URL check.

## Earlier findings
| ID | Title | Status |
|---|---|---|
| R-1 | CI had no explicit `permissions:` | fixed (`contents: read`), tested |
| R-2 | Actions pinned by tag | fixed (SHA pins), tested |
| R-3 | `/ingest` would be open if `API_KEY` empty | fixed: endpoint calls `authorize_ingest` (fails closed), tested over HTTP |
| R-4 | Indirect prompt injection via corpus | open (hypothesis), see LLM01 above |
