# Evals

Status: **TODO, nothing has run yet. No numbers are published.**

## What will be measured
| Metric | Definition | CI gate |
|---|---|---|
| hit rate@5 | share of questions where at least one gold chunk is in the top 5 | >= 0.80 |
| recall@5 | share of gold chunks recovered in the top 5, averaged over questions | >= 0.70 |
| faithfulness | share of answer sentences entailed by the cited chunks (local NLI cross-encoder) | >= 0.85 |
| faithfulness (LLM judge) | offline only, reported in the README | none |

Metric functions live in `src/rag_production/metrics.py`.

## Labeled set
`evals/labeled_set.jsonl`, one JSON object per line:
`{"id": "...", "question": "...", "relevant_chunk_ids": ["..."], "answerable": true}`.
To be written by hand from the FastAPI docs. It should cover answerable, partially
answerable and unanswerable questions.

## How to run (planned)
`rag eval --config evals/config.yaml`, which writes `reports/eval.json` and `reports/eval.md`.
