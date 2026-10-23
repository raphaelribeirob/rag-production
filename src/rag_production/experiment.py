"""Helpers for resumable retrieval experiments (tools/retrieval_experiment.py).

Answers are appended one JSON object per line and fsynced, so a killed run resumes from the file.
Decision rule (fixed 2026-10-23 before any test run): a config replaces the default only if, under
BOTH the small and the large NLI judge, (a) the paired bootstrap 95% CI of the per-question
faithfulness gain excludes 0 and (b) its worst seed beats the baseline's best seed.
"""

import json
import os
import statistics
from collections.abc import Iterable
from pathlib import Path

import numpy as np

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}  # t(0.975, df = n - 1)


def read_jsonl(path: str | Path) -> list[dict]:
    """Complete records only: a truncated last line (killed mid-write) is dropped."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def append_jsonl(path: str | Path, rec: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    good = read_jsonl(p)
    if p.exists() and len(p.read_text(encoding="utf-8").splitlines()) != len(good):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good), "utf-8")
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_interval(values: Iterable[float]) -> dict:
    """Mean, min-max and t-based 95% CI over independent runs (seeds)."""
    v = [float(x) for x in values]
    out = {"n": len(v), "mean": round(statistics.fmean(v), 4), "min": min(v), "max": max(v)}
    if len(v) < 2:
        return out | {"ci95": None}
    half = T95[len(v)] * statistics.stdev(v) / len(v) ** 0.5
    return out | {"ci95": [round(out["mean"] - half, 4), round(out["mean"] + half, 4)]}


def per_question_mean(runs: list[list[dict]]) -> dict[str, float]:
    """id -> mean score over runs (seeds); unscored (None) entries are skipped."""
    acc: dict[str, list[float]] = {}
    for items in runs:
        for it in items:
            if it["score"] is not None:
                acc.setdefault(it["id"], []).append(float(it["score"]))
    return {k: round(statistics.fmean(v), 4) for k, v in acc.items()}


def paired_bootstrap(
    base: dict[str, float], new: dict[str, float], iters: int = 10000, seed: int = 0
) -> dict:
    """Mean per-question difference new - base with a percentile bootstrap 95% CI."""
    ids = sorted(set(base) & set(new))
    d = np.array([new[i] - base[i] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(iters, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(ids), "diff": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}


def verdict(by_judge: dict[str, dict]) -> bool:
    """True only if every judge shows a gain outside the noise (see module docstring)."""
    return all(
        j["bootstrap"]["ci95"][0] > 0 and j["new"]["min"] > j["base"]["max"]
        for j in by_judge.values()
    )
