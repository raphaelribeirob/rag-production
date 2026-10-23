# Frozen question splits

Frozen 2026-10-22 (UTC-3), before any run on the dev set and before any prompt change.
`tests/test_eval_splits.py` fails if either file changes.

| sha256 and file | role |
|---|---|
| `a2310b4372f3e7df655d25c0fa9da530d83393e83624e2524758cb32753104d5` `evals/labeled_set.jsonl` | **test**: 57 questions, unchanged since the first eval; used once per final prompt |
| `f64eb5e43b99c808cecce34d1d585edcca9d58d0c22e288477a268e34793ac35` `evals/dev_set.jsonl` | **dev**: 23 NEW answerable questions; the only set used to iterate prompts |

Dev questions were written from page sections the test questions do not ask about (deprecation,
custom exception headers, Enum path params, JSON Lines, frontend fallback, ...), labeled by page,
before any model saw them. Prompt versions live in `src/rag_production/prompts.py`.
