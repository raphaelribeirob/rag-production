"""Dev/test question splits are frozen: hashes in evals/SPLITS.md must match the files."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    lines = (ROOT / "evals" / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def test_split_hashes_match_frozen_files():
    text = (ROOT / "evals" / "SPLITS.md").read_text(encoding="utf-8")
    pins = re.findall(r"`([0-9a-f]{64})`\s+`(evals/[\w.]+)`", text)
    assert {p for _, p in pins} == {"evals/labeled_set.jsonl", "evals/dev_set.jsonl"}
    for digest, path in pins:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_dev_questions_are_new_and_labeled():
    test, dev = _load("labeled_set.jsonl"), _load("dev_set.jsonl")
    assert len(dev) >= 20
    assert not {q["id"] for q in dev} & {q["id"] for q in test}
    test_q = {q["question"].lower() for q in test}
    assert all(q["question"].lower() not in test_q for q in dev)
    assert all(q["id"].startswith("d") and q["relevant_doc_ids"] for q in dev)
