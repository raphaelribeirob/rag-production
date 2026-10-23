"""Retrieval eval: Recall@5 and Hit@5 at document level over a labeled question set.

Usage: DATABASE_URL=... python -m rag_production.eval_runner [--limit N] [--out reports/eval.json]
Exits 1 if a gate fails. Faithfulness (NLI judge) is not implemented yet.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from rag_production.metrics import hit_at_k, recall_at_k

GATES = {"recall_at_5": 0.70, "hit_at_5": 0.80}


def load_labeled_set(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _docs(chunk_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for cid in chunk_ids:
        d = cid.split("#")[0]
        if d not in seen:
            seen.append(d)
    return seen


def evaluate(labeled: list[dict], retrieve: Callable[[str, int], list[str]], k: int = 5) -> dict:
    """retrieve(question, k) -> ranked chunk ids; relevance is judged per source document."""
    per_q = []
    for row in labeled:
        docs = _docs(retrieve(row["question"], k))[:k]
        rel = set(row["relevant_doc_ids"])
        per_q.append(
            {"id": row["id"], "hit": hit_at_k(docs, rel, k), "recall": recall_at_k(docs, rel, k)}
        )
    n = len(per_q)
    return {
        "n": n,
        "k": k,
        "hit_at_5": round(sum(q["hit"] for q in per_q) / n, 4),
        "recall_at_5": round(sum(q["recall"] for q in per_q) / n, 4),
        "misses": [q["id"] for q in per_q if q["hit"] == 0.0],
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised in CI eval step
    from rag_production.config import load_settings
    from rag_production.embeddings import SentenceTransformerEmbedder
    from rag_production.ingest import ingest_dir
    from rag_production.store import VectorStore

    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="evals/labeled_set.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    ap.add_argument("--out", default="reports/eval.json")
    args = ap.parse_args(argv)

    settings = load_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.database_url, dim=embedder.dim, table="chunks_eval")
    store.migrate()
    store.clear()
    ingest = ingest_dir(settings.corpus_dir, store, embedder)
    labeled = load_labeled_set(args.labeled)
    if args.limit:
        labeled = labeled[: args.limit]
    k = max(5, settings.top_k)

    def retrieve(q: str, n: int) -> list[str]:
        # over-fetch chunks so that 5 distinct documents can be ranked
        return [h.chunk_id for h in store.search(embedder.embed([q])[0], k=n * 4)]

    report = evaluate(labeled, retrieve, k=5)
    report.update({"model": embedder.name, "corpus": ingest, "gates": GATES, "top_k": k})
    report["faithfulness"] = "not implemented yet (NLI judge planned)"
    failed = [m for m, g in GATES.items() if report[m] < g]
    report["passed"] = not failed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    store.close()
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
