# Prompt choice (dev only, decided 2026-10-22 23:50 UTC-3, before any test run with a new prompt)

Dev split `evals/dev_set.jsonl` (23 answerable questions), Qwen2.5-7B-Instruct Q4_K_M via
llama-cpp-python 0.3.36, judge cross-encoder/nli-deberta-v3-small (unchanged).

| prompt | faithfulness | scored | negative control | supported | hallucination | evasive | paraphrase |
|---|---|---|---|---|---|---|---|
| v1 (baseline) | 0.8623 | 23 | 0.1304 | 0.7391 | 0.1739 | 0.0435 | 0.0435 |
| v2 | 0.4000 | 5 of 23 | 0.0000 | 0.4 | 0.2 | 0.0 | 0.4 |
| v3 | 0.5000 | 14 of 23 | 0.0238 | 0.4286 | 0.1429 | 0.0 | 0.4286 |
| v4 | 0.7029 | 23 | 0.1449 | 0.5217 | 0.2609 | 0.0435 | 0.1739 |
| v5 | 0.8333 | 23 | 0.1812 | 0.6957 | 0.3043 | 0.0 | 0.0 |

v1 0.8623 is the rebuilt-environment run (the box was restored mid-task and llama-cpp-python was
rebuilt; the first dev v1 run gave 0.8696). v2 ran before the restore; only its summary numbers
survived (raw files lost). v2/v3 ask for [n] citations: the model often puts them after the final
period, and the unchanged judge drops such sentences as unscorable, so citations were dropped.

Decision: no new prompt beat v1 on dev. v5 (best new prompt on dev) is the single candidate run
once on test; the default stays v1 unless test says otherwise.
