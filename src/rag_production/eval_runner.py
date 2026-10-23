"""Retrieval eval: Recall@5 and Hit@5 at document level over a labeled question set.

Usage: DATABASE_URL=... python -m rag_production.eval_runner [--limit N] [--out reports/eval.json]
Exits 1 if a gate fails. Faithfulness: local NLI judge over answerable, non-abstained questions.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_production.metrics import hit_at_k, recall_at_k

GATES = {"recall_at_5": 0.70, "hit_at_5": 0.80, "faithfulness": 0.85}


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


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "hit_at_5": None, "recall_at_5": None}
    return {
        "n": n,
        "hit_at_5": round(sum(q["hit"] for q in rows) / n, 4),
        "recall_at_5": round(sum(q["recall"] for q in rows) / n, 4),
    }


def _rate(rows: list[dict]) -> float | None:
    return round(sum(q["abstained"] for q in rows) / len(rows), 4) if rows else None


def evaluate(
    labeled: list[dict],
    retrieve: Callable[[str, int], list[tuple[str, float]]],
    k: int = 5,
    abstain_threshold: float = 0.40,
) -> dict:
    """retrieve(question, n) -> ranked (chunk_id, cosine). Relevance is judged per document.

    Retrieval metrics use answerable questions only; abstention = top-1 score < threshold.
    """
    per_q = []
    for row in labeled:
        ranked = retrieve(row["question"], k)
        docs = _docs([cid for cid, _ in ranked])[:k]
        rel = set(row["relevant_doc_ids"])
        top = ranked[0][1] if ranked else 0.0
        per_q.append(
            {
                "id": row["id"],
                "category": row.get("category", "single" if rel else "unanswerable"),
                "answerable": bool(rel),
                "hit": hit_at_k(docs, rel, k) if rel else 0.0,
                "recall": recall_at_k(docs, rel, k) if rel else 0.0,
                "abstained": top < abstain_threshold,
                "top_score": round(top, 4),
            }
        )
    ans = [q for q in per_q if q["answerable"]]
    unans = [q for q in per_q if not q["answerable"]]
    cats = sorted({q["category"] for q in ans})
    overall = _agg(ans)
    return {
        "n": len(per_q),
        "k": k,
        "hit_at_5": overall["hit_at_5"],
        "recall_at_5": overall["recall_at_5"],
        "answerable": overall,
        "by_category": {c: _agg([q for q in ans if q["category"] == c]) for c in cats},
        "abstention": {
            "threshold": abstain_threshold,
            "unanswerable_n": len(unans),
            "unanswerable_abstain_rate": _rate(unans),
            "answerable_false_abstain_rate": _rate(ans),
            "overall_abstain_rate": _rate(per_q),
        },
        "misses": [q["id"] for q in ans if q["hit"] == 0.0],
        "unanswerable_top_scores": {q["id"]: q["top_score"] for q in unans},
    }


def gate_failures(report: dict) -> list[str]:
    """Metrics below their gate; a missing (None) metric fails its gate."""
    return [m for m, g in GATES.items() if report.get(m) is None or report[m] < g]


def latency_summary(ms: list[float]) -> dict | None:
    """Mean / p50 / p95 / max of per-query latencies (linear interpolation)."""
    if not ms:
        return None
    a = np.asarray(ms, dtype=float)
    return {
        "n": len(ms),
        "mean_ms": round(float(a.mean()), 1),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "max_ms": round(float(a.max()), 1),
    }


def answer_for(question: str, texts: list[str], provider) -> tuple[str, str]:
    """Generated answer when a provider exists, else the extractive answer (top chunk)."""
    if provider is None:
        return texts[0], "extractive (top chunk verbatim)"
    return provider.generate(question, texts), f"generated ({provider.name})"


def answer_record(row: dict, chunk_ids: list[str], texts: list[str], answer: str, ms: float):
    """Saved per-answer record: ids, retrieved docs (top-5 chunks), gold hit, latency, text."""
    docs = _docs(chunk_ids)
    return {
        "id": row["id"],
        "question": row["question"],
        "relevant_doc_ids": row["relevant_doc_ids"],
        "doc_ids": docs,
        "gold_retrieved": bool(set(docs) & set(row["relevant_doc_ids"])),
        "latency_ms": round(ms, 1),
        "answer": answer,
        "contexts": texts,
    }


def _faithfulness(labeled, store, embedder, settings, args) -> dict:  # pragma: no cover
    from rag_production.faithfulness import NLIJudge, evaluate_faithfulness
    from rag_production.providers import get_provider

    provider = get_provider(settings)
    items, source, gen_ms = [], "none", []
    for row in labeled:
        if not row["relevant_doc_ids"]:
            continue
        hits = store.search(embedder.embed([row["question"]])[0], k=5)
        if not hits or hits[0].score < settings.abstain_threshold:
            continue
        if args.faithfulness_limit and len(items) >= args.faithfulness_limit:
            break
        texts = [h.text for h in hits]
        t0 = time.perf_counter()
        answer, source = answer_for(row["question"], texts, provider)
        gen_ms.append((time.perf_counter() - t0) * 1000)
        items.append(answer_record(row, [h.chunk_id for h in hits], texts, answer, gen_ms[-1]))
    judge = NLIJudge()
    result = evaluate_faithfulness(items, judge)
    result.update({"judge": judge.name, "answer_source": source})
    if provider is not None and hasattr(provider, "prompt_version"):
        result["prompt_version"] = provider.prompt_version
    result["answer_latency"] = latency_summary(gen_ms)  # generation only (0 for extractive)
    if args.save_answers:
        Path(args.save_answers).write_text(json.dumps(items, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised in CI eval step
    from rag_production.config import load_settings
    from rag_production.embeddings import SentenceTransformerEmbedder
    from rag_production.ingest import ingest_dir
    from rag_production.store import VectorStore

    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="evals/labeled_set.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    ap.add_argument("--out", default="reports/eval.json")
    ap.add_argument("--faithfulness-limit", type=int, default=0, help="judge only N answers")
    ap.add_argument("--save-answers", default="", help="write judged answers + contexts (JSON)")
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
        hits = store.search(embedder.embed([q])[0], k=n * 4)
        return [(h.chunk_id, h.score) for h in hits]

    report = evaluate(labeled, retrieve, k=5, abstain_threshold=settings.abstain_threshold)
    report.update({"model": embedder.name, "corpus": ingest, "gates": GATES, "top_k": k})
    report["faithfulness_eval"] = _faithfulness(labeled, store, embedder, settings, args)
    report["faithfulness"] = report["faithfulness_eval"]["faithfulness"]
    failed = gate_failures(report)
    report["passed"] = not failed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    store.close()
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
