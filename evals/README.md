# Evals

Status: retrieval eval and NLI faithfulness are implemented and gated in CI (CI scores the mock).
A real local LLM (Qwen2.5-7B-Instruct Q4_K_M) was measured once on the dev box: faithfulness 0.647.

## Metrics
| Metric | Definition | CI gate |
|---|---|---|
| hit@5 | share of questions where at least one relevant document is among the top 5 distinct documents retrieved | >= 0.80 |
| recall@5 | share of relevant documents among the top 5 distinct documents, averaged over questions | >= 0.70 (stretch 0.80) |
| faithfulness | share of answer sentences entailed by cited chunks (local NLI `cross-encoder/nli-deberta-v3-small`) | >= 0.85 |

Abstention: `/query` and the eval abstain when the top-1 chunk cosine is below
`ABSTAIN_THRESHOLD` (default 0.40, fixed before the first run with unanswerable questions and
not tuned since). Abstention rates are reported, **not gated**.

Retrieval metrics (and the gates) use answerable questions only. Relevance is labeled per source document (`relevant_doc_ids`), not per chunk, so labels survive
chunker changes. The runner over-fetches 20 chunks and ranks distinct documents.

## Labeled set
`evals/labeled_set.jsonl` (57 questions), each with a `category`:
- `single` (35): one relevant tutorial page.
- `multi` (12): 2 or 3 relevant pages; recall needs all of them.
- `unanswerable` (10): 7 off-corpus FastAPI topics (WebSockets, Kubernetes, Docker, rate
  limiting, Gunicorn, Jinja2, JWT) and 3 off-domain questions. The right behavior is to abstain.

All questions were written from page topics before the eval ran, and committed alone
(`2c6f446` for multi/unanswerable) before the run. They are not tuned to results.

## Run
```bash
python tools/fetch_corpus.py && python tools/migrate.py
python -m rag_production.eval_runner --out reports/eval.json   # exits 1 if a gate fails
```
Use `--limit N` / `--faithfulness-limit N` for a quick sample. Retrieval takes about 15 s on CPU;
with the NLI judge the full run took about 2 to 3 min on the dev box and about 9 min in CI.
CI runs the full set, not a sample.

## Results (2026-10-22, all-MiniLM-L6-v2, 442 chunks, threshold 0.40)
| Category | n | Hit@5 | Recall@5 | Abstained |
|---|---|---|---|---|
| single | 35 | 1.00 | 0.986 | 0% (false abstention) |
| multi | 12 | 1.00 | 0.889 | 0% (false abstention) |
| answerable overall (gated) | 47 | 1.00 | 0.961 | 0% |
| unanswerable | 10 | n/a | n/a | 70% (correct abstention) |
| all questions | 57 | - | - | 12.3% |

Failures: 3 of 10 unanswerable questions were answered instead of abstaining: WebSockets
(top score 0.49), Kubernetes (0.66) and Dockerfile (0.64). Near-domain questions score above the
threshold because the corpus talks about servers and deployment in passing. All 3 off-domain
questions and 4 of 7 off-corpus FastAPI questions were abstained.

## Faithfulness (NLI)
Each answer sentence (prose only: code, headings, tables and list fragments are skipped) counts as
supported when `P(entailment) >= 0.5` for at least one premise unit from the top-5 retrieved chunks.
Premise units are each chunk **and each sentence of each chunk** (SummaC-ZS style). Score = share of
supported sentences, averaged over answers. A negative control re-scores each answer against
another question's chunks; it must stay low, or the judge is not discriminating.

Answer source: the generated answer when `LLM_PROVIDER` selects a provider, otherwise the
extractive answer (top chunk verbatim). `reports/eval.json` records it in
`faithfulness_eval.answer_source`. **CI sets `LLM_PROVIDER=mock`**, so CI measures the mock.

Measured 2026-10-22 (answerable, non-abstained questions):
| Answer source | n | Faithfulness | Negative control | Where |
|---|---|---|---|---|
| generated (mock) | 47 | 1.000 | 0.085 | CI + local |
| extractive (top chunk) | 45 | 0.993 | 0.055 | local (CI before the providers step) |
| generated (llamacpp: Qwen2.5-7B-Instruct Q4_K_M) | 46 | **0.647** (fails 0.85) | 0.150 | local only, see below |

n differs because 2 top chunks have no prose sentence the judge scores (code or lists only).
- Mock fix, disclosed: the first mock run scored **0.830**. The mock copied lead-in sentences
  ending in ":" (they introduce code), and joining two of them merged sentences across chunks into
  text that is in no chunk. The mock now copies only sentences ending in . ! or ?. Gate unchanged.
- Method change, disclosed: the first version used whole chunks as premises only and scored
  **0.776** (negative control 0.035), failing the gate even though the answers are verbatim copies
  of the context. Debugging showed the small NLI model gives low entailment for a sentence against
  a multi-sentence premise that contains it (e.g. 0.45 against a 2-sentence premise vs 0.99 alone).
  That is a measurement bug, so sentence-level premises were added. The gate was not changed.
- Caveat: extractive and mock answers are copied from the context, so high scores are expected.
  They check the judge and the code path, not a generator. The real-model number is below.

## Real LLM faithfulness (local, 2026-10-22)
Measured on the dev box only. **Not run in CI and not gated in CI** (CI still gates the mock).

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct-GGUF`, file `qwen2.5-7b-instruct-q4_k_m` (2 shards), **Q4_K_M** |
| Runtime | `llama-cpp-python` 0.3.36, in-process, CPU only (8 vCPU Xeon, 15 GB RAM), `n_ctx` 4096, temperature 0, seed 0, `max_tokens` 200 |
| Integrity | both shards' sha256 checked against the Hugging Face LFS hashes before loading |
| Faithfulness (answerable, non-abstained, n=46 scored) | **0.647**: below the 0.85 gate; the run exits 1 |
| Negative control | 0.150 |
| Per answer | 21 of 46 fully supported, 7 scored 0, 18 partly supported |
| Generation latency per query (n=47) | mean 17.3 s, p50 15.5 s, p95 29.7 s, max 39.0 s |
| Whole run (retrieval, generation, NLI) | 16 min 12 s wall clock |
| Retrieval (same run) | hit@5 1.00, recall@5 0.961, abstention unchanged (70% unanswerable, 12.3% overall) |

Faithfulness covers the 47 answerable questions, as in CI. 7B was fast enough for them, so 3B
and 1.5B were not needed. One answer has no sentence the judge scores (n=46).

Exact command (summary JSON committed as
`evals/results/2026-10-22_llamacpp_qwen2.5-7b-instruct-q4_k_m.json`):
```bash
pip install -e ".[local-llm]"   # llama-cpp-python; optional, never installed in CI
LLM_PROVIDER=llamacpp \
LLAMACPP_MODEL_PATH=/home/box/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
python -m rag_production.eval_runner --out reports/eval_llamacpp_qwen2.5-7b.json \
  --save-answers reports/answers_llamacpp_qwen2.5-7b.json
```

Reading the low-scoring answers (no tuning was done after seeing them):
- Some answers are wrong: for example, the custom-header answer invents `header.user_agent`, and
  the UUID/Decimal answer says Decimals "would typically be represented as floats", which is not in
  the retrieved chunks (the full page does say Decimal is handled like float, but that chunk was not
  retrieved). Unsupported by the context either way.
- Some are hedges or non-answers ("the context does not provide specific details..."). The judge
  counts these as unsupported. That is arguably right for a grounded system, but it is not a
  hallucination of facts.
- Answers that paraphrase or join facts from several chunks score lower than verbatim copies,
  because the small NLI judge checks each sentence against one premise unit at a time.
So 0.647 mixes real errors, hedges and judge limits. It is not a pure hallucination rate. The prompt, judge and gate were **not** changed in response to this number.

How the model was obtained: route (a), the official Ollama Linux binary from GitHub Releases, was
downloaded and its published sha256 verified. But `ollama pull` was refused on this box with
`redirect target not allowed: <bucket>.r2.cloudflarestorage.com resolves to non-public 198.18.0.1`
(the box's DNS maps the model CDN to a non-public address and Ollama's guard refuses it). That
was not worked around. Route (b), `llama-cpp-python` plus the official Qwen GGUF, was used instead.

Abstention threshold: left at 0.40 on purpose. It was not tuned on the test set, and no dev split
was made, so the test-set abstention numbers above are unchanged.

## Failure taxonomy and prompt v5 (2026-10-22/23, local CPU)
Same model, runtime and judge as above. Not in CI. Gate (0.85), CI judge and threshold (0.40) unchanged.

**Classifier** (`src/rag_production/taxonomy.py`, rules frozen in c7d79a1 before use, tested in
`tests/test_taxonomy.py`). Per sentence, first match wins: entailment >= 0.5 supported; refusal
phrase is evasive if a gold page is in the top 5, else justified refusal; contradiction >= 0.5
hallucination; a code identifier not in the context hallucination; content-token overlap >= 0.75
paraphrase; else hallucination. Answer label = worst sentence.

**Hand audit** (`evals/audit/hand_labels_before.json`): 20 BEFORE answers labeled blind by the agent
(Grok Bot), not by a human. Classifier agreement 13/20 = **0.65** (also 0.65 on 3 classes). 5 answers
labeled grounded were called hallucination (paraphrases the small judge does not entail), 1 evasive
was called hallucination, 1 hallucination was called supported. So the classifier over-calls
hallucination; treat its hallucination rate as an upper bound.

**Splits**: `evals/SPLITS.md` freezes the dev (23 answerable) and test files by sha256
(`tests/test_eval_splits.py`). Prompts were iterated on dev only (`evals/results/dev/DECISION.md`):
citation prompts v2/v3 failed (the unchanged judge cannot score sentences ending in "[n]"), v4 0.70,
v5 0.83, v1 0.86. v5 was picked as the single test candidate before any new-prompt test run, and run
on test **once**.

**Environment drift**: the box was restored mid-task and llama-cpp-python was rebuilt from source.
The same v1 prompt on test then scored 0.588 instead of 0.647, so a v1 control was rerun on the
rebuilt env. The fair BEFORE vs AFTER pair is v1-rebuilt vs v5 (same build). Re-scoring the
committed BEFORE answers reproduced 0.6467 and identical labels, so the judge did not drift; the generator did.

| test (47 answerable) | BEFORE v1 (pre-restore) | v1 control (rebuilt) | AFTER v5 (rebuilt) |
|---|---|---|---|
| faithfulness (gate 0.85) | 0.647 (n=46) | 0.588 | **0.605** |
| negative control | 0.150 | 0.099 | 0.153 |
| supported | 0.457 | 0.383 | 0.447 |
| hallucination (upper bound) | 0.457 | 0.468 | 0.426 |
| evasive refusal | 0.000 | 0.021 | 0.021 |
| unrecognized paraphrase | 0.087 | 0.128 | 0.106 |
| recall@5 / hit@5 | 0.961 / 1.00 | 0.961 / 1.00 | 0.961 / 1.00 |
| latency mean / p50 / p95 / max (s) | 17.3 / 15.5 / 29.7 / 39.0 | 21.1 / 17.1 / 42.1 / 99.7 | 17.3 / 15.4 / 33.6 / 53.2 |

v5 beats the same-build control by 0.017 faithfulness but is below the original 0.647 and far below
0.85. With 47 questions and a 0.65-agreement classifier that gap is within noise. **Default prompt
stays v1.** Latency on the v1 control was likely inflated by other jobs on the box.

Files: `evals/results/2026-10-22_test_eval_{v1_rebuilt,v5}.json`,
`evals/results/2026-10-22_taxonomy_{before_v1,test_v1_rebuilt,after_v5}.json`.
Commands (`M` = GGUF path above):
```bash
LLM_PROVIDER=llamacpp LLAMACPP_MODEL_PATH=$M LLM_PROMPT=v5 PYTHONPATH=src \
  python -m rag_production.eval_runner --out reports/test_v5.json --save-answers reports/test_answers_v5.json
PYTHONPATH=src python tools/taxonomy_report.py --answers reports/test_answers_v5.json \
  --judge cross-encoder/nli-deberta-v3-small --out evals/results/2026-10-22_taxonomy_after_v5.json
# BEFORE raw answers were lost in the restore: re-pair committed answers with re-retrieved contexts
PYTHONPATH=src python tools/taxonomy_report.py --rebuild \
  --answers evals/results/2026-10-22_taxonomy_before_v1.json --hand evals/audit/hand_labels_before.json \
  --judge cross-encoder/nli-deberta-v3-small --out reports/rebuild_check_small.json
```
Dev runs use `--labeled evals/dev_set.jsonl` (see `evals/results/dev/DECISION.md`).

## Larger NLI judge, offline comparison only (2026-10-23, local CPU)
Same answers and contexts as above, re-scored with bigger cross-encoders from Hugging Face. **For
comparison only: the CI judge (small) and the 0.85 gate do not change, and 0.647 stays the baseline.**
Revisions: small `fa2804872c3b4bd748f38c0185cc85775361e735`, base
`6c749ce3425cd33b46d187e45b92bbf96ee12ec7`, large `bab4bc7178836f731dcfd18c06ca9def0a137712`.
CPU time per 47-answer set: base about 6 min, large about 20 to 30 min (8 vCPU).

| answers (test, 47) | judge | n | faithfulness | neg. control | supported | hallucination | evasive | paraphrase | hand agreement |
|---|---|---|---|---|---|---|---|---|---|
| BEFORE v1 (the 0.647 run) | small | 46 | 0.647 | 0.150 | 0.457 | 0.457 | 0.000 | 0.087 | 13/20 (0.65) |
| | base | 46 | 0.511 | 0.074 | 0.326 | 0.609 | 0.000 | 0.065 | 14/20 (0.70) |
| | large | 46 | 0.639 | 0.198 | 0.457 | 0.478 | 0.022 | 0.043 | 14/20 (0.70) |
| v1 control (rebuilt) | small | 47 | 0.588 | 0.099 | 0.383 | 0.468 | 0.021 | 0.128 | - |
| | base | 47 | 0.489 | 0.069 | 0.277 | 0.638 | 0.000 | 0.085 | - |
| | large | 47 | 0.598 | 0.180 | 0.404 | 0.468 | 0.043 | 0.085 | - |
| AFTER v5 (rebuilt) | small | 47 | 0.605 | 0.152 | 0.447 | 0.425 | 0.021 | 0.106 | - |
| | base | 47 | 0.477 | 0.028 | 0.255 | 0.617 | 0.021 | 0.106 | - |
| | large | 47 | 0.683 | 0.135 | 0.468 | 0.425 | 0.043 | 0.064 | - |

Justified refusal is 0.000 everywhere. Hand agreement is the 4-class rate against
`evals/audit/hand_labels_before.json` (3-class is the same). Reading:
- 0.647 roughly holds under the large judge (0.639) but not under base (0.511). Base is stricter
  than small, not more lenient, so "the small judge under-credits paraphrase" is not the whole story.
- Every judge keeps the answers far below 0.85. The negative control stays low (0.03 to 0.20).
- The judges disagree on v5 vs the v1 control: small +0.017, base -0.012, large +0.085. The ranking
  depends on the judge, so with 47 questions the prompt difference is not established. Default prompt stays v1.
- Classifier agreement with the hand audit rises from 13/20 to 14/20 with base or large; it still
  over-calls hallucination (4 hand-grounded answers labeled hallucination).

Files: `evals/results/2026-10-22_judge_{base,large}_{before_v1,test_v1_rebuilt,after_v5}.json`.
Commands (`J` = `cross-encoder/nli-deberta-v3-base` or `-large`, `S` = `base` or `large`):
```bash
PYTHONPATH=src python tools/taxonomy_report.py --rebuild \
  --answers evals/results/2026-10-22_taxonomy_before_v1.json --hand evals/audit/hand_labels_before.json \
  --judge $J --out evals/results/2026-10-22_judge_${S}_before_v1.json
PYTHONPATH=src python tools/taxonomy_report.py --answers reports/test_answers_v1_rebuilt.json \
  --judge $J --out evals/results/2026-10-22_judge_${S}_test_v1_rebuilt.json
PYTHONPATH=src python tools/taxonomy_report.py --answers reports/test_answers_v5.json \
  --judge $J --out evals/results/2026-10-22_judge_${S}_after_v5.json
```
