# Evals

Status: retrieval eval and NLI faithfulness are implemented and gated in CI.

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
| generated (ollama:qwen2.5:7b) | - | not measured | - | Ollama not run |

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
  They check the judge and the code path, not a generator. A real number needs Ollama.
