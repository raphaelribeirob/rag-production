import json

from rag_production.eval_runner import evaluate, load_labeled_set


def test_evaluate_doc_level_metrics():
    labeled = [
        {"id": "q1", "question": "a", "relevant_doc_ids": ["a.md"]},
        {"id": "q2", "question": "b", "relevant_doc_ids": ["b.md", "c.md"]},
    ]
    ranked = {"a": ["a.md#0", "a.md#1", "x.md#0"], "b": ["b.md#2", "y.md#0"]}
    report = evaluate(labeled, lambda q, k: ranked[q][:k], k=5)
    assert report["n"] == 2
    assert report["hit_at_5"] == 1.0
    assert report["recall_at_5"] == 0.75
    json.dumps(report)


def test_labeled_set_is_well_formed():
    rows = load_labeled_set("evals/labeled_set.jsonl")
    assert len(rows) >= 30
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["question"] and r["relevant_doc_ids"] for r in rows)
