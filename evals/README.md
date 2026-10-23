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
with the NLI judge the full run took about 3 min on the dev box. CI runs the full set.

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

Measured 2026-10-22 on 45 answerable, non-abstained questions:
- Answer source: **extractive** (top chunk verbatim). No LLM generation exists yet.
- faithfulness **0.993**, negative control **0.055**, gate >= 0.85: passes.
- Method change, disclosed: the first version used whole chunks as premises only and scored
  **0.776** (negative control 0.035), failing the gate even though the answers are verbatim copies
  of the context. Debugging showed the small NLI model gives low entailment for a sentence against
  a multi-sentence premise that contains it (e.g. 0.45 against a 2-sentence premise vs 0.99 alone).
  That is a measurement bug, so sentence-level premises were added. The gate was not changed.
- Caveat: verbatim extractive answers make a high score expected; this checks the judge, not a
  generator. It becomes meaningful once LLM providers are added.
