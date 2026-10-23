# Evals

Status: retrieval eval implemented and gated in CI. Faithfulness: **not implemented yet**.

## Metrics
| Metric | Definition | CI gate |
|---|---|---|
| hit@5 | share of questions where at least one relevant document is among the top 5 distinct documents retrieved | >= 0.80 |
| recall@5 | share of relevant documents among the top 5 distinct documents, averaged over questions | >= 0.70 (stretch 0.80) |
| faithfulness | share of answer sentences entailed by cited chunks (local NLI `cross-encoder/nli-deberta-v3-small`) | >= 0.85, planned |

Relevance is labeled per source document (`relevant_doc_ids`), not per chunk, so labels survive
chunker changes. The runner over-fetches 20 chunks and ranks distinct documents.

## Labeled set
`evals/labeled_set.jsonl`: 35 hand-written questions over the 40 tutorial pages listed in
`data/SOURCES.md`, written from page topics before the first run (not tuned on results).
All are answerable. TODO: add unanswerable and multi-page questions; the current set is easy.

## Run
```bash
python tools/fetch_corpus.py && python tools/migrate.py
python -m rag_production.eval_runner --out reports/eval.json   # exits 1 if a gate fails
```
Use `--limit N` for a quick sample. The full set takes about 15 s on CPU, so CI runs all of it.

## Results (2026-10-22, all-MiniLM-L6-v2, 442 chunks)
hit@5 = 1.00, recall@5 = 0.9857, n = 35, no misses.
