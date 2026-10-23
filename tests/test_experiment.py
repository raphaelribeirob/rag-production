"""Resumable experiment helpers: JSONL append/resume, run intervals, paired bootstrap, verdict."""

import json

import pytest

from rag_production.experiment import (
    append_jsonl,
    paired_bootstrap,
    per_question_mean,
    read_jsonl,
    run_interval,
    verdict,
)


def test_jsonl_append_and_resume_drops_a_truncated_last_line(tmp_path):
    p = tmp_path / "a.jsonl"
    assert read_jsonl(p) == []
    append_jsonl(p, {"id": "q1", "answer": "x"})
    append_jsonl(p, {"id": "q2", "answer": "y"})
    with p.open("a") as fh:
        fh.write('{"id": "q3", "ans')  # killed mid-write
    assert [r["id"] for r in read_jsonl(p)] == ["q1", "q2"]
    append_jsonl(p, {"id": "q3", "answer": "z"})  # resume rewrites cleanly
    assert [r["id"] for r in read_jsonl(p)] == ["q1", "q2", "q3"]
    assert all(json.loads(line) for line in p.read_text().splitlines())


def test_run_interval_mean_range_and_t_ci():
    r = run_interval([0.5, 0.6, 0.7])
    assert r["n"] == 3 and r["mean"] == pytest.approx(0.6)
    assert (r["min"], r["max"]) == (0.5, 0.7)
    lo, hi = r["ci95"]  # t(0.975, df=2) = 4.303, sd = 0.1, se = 0.0577
    assert lo == pytest.approx(0.6 - 4.303 * 0.1 / 3**0.5, abs=1e-4)
    assert hi == pytest.approx(0.6 + 4.303 * 0.1 / 3**0.5, abs=1e-4)
    assert run_interval([0.4])["ci95"] is None


def test_per_question_mean_averages_seeds_and_skips_unscored():
    runs = [
        [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.0}],
        [{"id": "a", "score": 0.5}, {"id": "b", "score": None}],
    ]
    assert per_question_mean(runs) == {"a": 0.75, "b": 0.0}


def test_paired_bootstrap_is_seeded_and_detects_a_clear_shift():
    base = {f"q{i}": 0.5 for i in range(30)}
    better = {f"q{i}": 0.5 + (0.2 if i % 2 else 0.1) for i in range(30)}
    r1, r2 = paired_bootstrap(base, better), paired_bootstrap(base, better)
    assert r1 == r2 and r1["n"] == 30
    assert r1["diff"] == pytest.approx(0.15) and r1["ci95"][0] > 0
    same = paired_bootstrap(base, dict(base))
    assert same["ci95"] == [0.0, 0.0]


def test_verdict_needs_both_judges_outside_noise():
    clear = {"bootstrap": {"ci95": [0.02, 0.1]}, "base": {"max": 0.6}, "new": {"min": 0.65}}
    overlap = {"bootstrap": {"ci95": [0.02, 0.1]}, "base": {"max": 0.7}, "new": {"min": 0.65}}
    noisy = {"bootstrap": {"ci95": [-0.01, 0.1]}, "base": {"max": 0.6}, "new": {"min": 0.65}}
    assert verdict({"small": clear, "large": clear}) is True
    assert verdict({"small": clear, "large": overlap}) is False
    assert verdict({"small": noisy, "large": clear}) is False
