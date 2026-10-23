import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("tr", ROOT / "tools" / "taxonomy_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rebuild_recovers_contexts_for_answers_of_a_committed_report():
    labeled = [
        {"id": "q1", "question": "Q1?", "relevant_doc_ids": ["a.md"]},
        {"id": "q2", "question": "Q2?", "relevant_doc_ids": ["z.md"]},
    ]
    report_items = [{"id": "q2", "answer": "A2."}, {"id": "q1", "answer": "A1."}]

    def search(question):
        doc = "a.md" if question == "Q1?" else "b.md"
        return [SimpleNamespace(chunk_id=f"{doc}#0", text=f"ctx for {question}")]

    out = _tool().rebuild(report_items, labeled, search)
    assert [r["id"] for r in out] == ["q2", "q1"]
    assert out[0]["contexts"] == ["ctx for Q2?"] and out[0]["gold_retrieved"] is False
    assert out[1]["answer"] == "A1." and out[1]["gold_retrieved"] is True
