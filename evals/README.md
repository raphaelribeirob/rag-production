# Evals

Status: retrieval eval implemented and gated in CI. Faithfulness: **not implemented yet**.

## Metrics
| Metric | Definition | CI gate |
|---|---|---|
| hit@5 | share of questions where at least one relevant document is among the top 5 distinct documents retrieved | >= 0.80 |
| recall@5 | share of relevant documents among the top 5 distinct documents, averaged over questions | >= 0.70 (stretch 0.80) |
| faithfulness | share of answer sentences entailed by cited chunks (local NLI `cross-encoder/nli-deberta-v3-small`) | >= 0.85, planned |

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
Use `--limit N` for a quick sample. The full set takes about 15 s on CPU, so CI runs all of it.

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
