import json

from rag_production.eval_runner import evaluate, load_labeled_set

LABELED = [
    {"id": "s1", "question": "a", "relevant_doc_ids": ["a.md"], "category": "single"},
    {"id": "m1", "question": "b", "relevant_doc_ids": ["b.md", "c.md"], "category": "multi"},
    {"id": "u1", "question": "z", "relevant_doc_ids": [], "category": "unanswerable"},
    {"id": "u2", "question": "y", "relevant_doc_ids": [], "category": "unanswerable"},
]
RANKED = {
    "a": [("a.md#0", 0.8), ("a.md#1", 0.7), ("x.md#0", 0.5)],
    "b": [("b.md#2", 0.6), ("y.md#0", 0.5)],
    "z": [("x.md#0", 0.2)],  # below threshold -> abstain (correct)
    "y": [("x.md#1", 0.9)],  # above threshold -> answers (wrong)
}


def _report():
    return evaluate(LABELED, lambda q, k: RANKED[q][:k], k=5, abstain_threshold=0.4)


def test_retrieval_metrics_by_category_and_overall():
    r = _report()
    assert r["by_category"]["single"] == {"n": 1, "hit_at_5": 1.0, "recall_at_5": 1.0}
    assert r["by_category"]["multi"] == {"n": 1, "hit_at_5": 1.0, "recall_at_5": 0.5}
    assert r["answerable"] == {"n": 2, "hit_at_5": 1.0, "recall_at_5": 0.75}
    json.dumps(r)


def test_abstention_rates():
    r = _report()
    assert r["abstention"]["threshold"] == 0.4
    assert r["abstention"]["unanswerable_abstain_rate"] == 0.5
    assert r["abstention"]["answerable_false_abstain_rate"] == 0.0
    assert r["abstention"]["overall_abstain_rate"] == 0.25


def test_labeled_set_is_well_formed():
    rows = load_labeled_set("evals/labeled_set.jsonl")
    assert len({r["id"] for r in rows}) == len(rows)
    cats = [r["category"] for r in rows]
    assert cats.count("single") >= 30
    assert cats.count("multi") >= 10
    assert cats.count("unanswerable") >= 10
    for r in rows:
        assert r["question"]
        assert r["answerable"] == bool(r["relevant_doc_ids"])
        if r["category"] == "multi":
            assert len(r["relevant_doc_ids"]) >= 2


def test_gate_failures_include_faithfulness_and_missing_metrics():
    from rag_production.eval_runner import GATES, gate_failures

    assert GATES["faithfulness"] == 0.85 and GATES["recall_at_5"] == 0.70
    ok = {"recall_at_5": 0.9, "hit_at_5": 0.9, "faithfulness": 0.9}
    assert gate_failures(ok) == []
    assert gate_failures({**ok, "faithfulness": 0.84}) == ["faithfulness"]
    assert gate_failures({**ok, "faithfulness": None}) == ["faithfulness"]


def test_faithfulness_answer_is_generated_when_provider_exists_else_extractive():
    from rag_production.eval_runner import answer_for
    from rag_production.providers import MockProvider

    texts = ["H\n\nFirst chunk sentence here. More text.", "H2\n\nSecond chunk sentence here."]
    assert answer_for("q", texts, None) == (texts[0], "extractive (top chunk verbatim)")
    ans, source = answer_for("q", texts, MockProvider())
    assert ans == "First chunk sentence here. Second chunk sentence here."
    assert source == "generated (mock)"
