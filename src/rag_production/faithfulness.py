"""Faithfulness: share of answer sentences entailed by the cited chunks (local NLI judge).

Judge: cross-encoder/nli-deberta-v3-small on CPU ($0, deterministic). A sentence counts as
supported if P(entailment | premise, sentence) >= threshold for at least one premise unit, where
units are each cited chunk plus each of its sentences (SummaC-ZS style: small NLI models handle
single-sentence premises far better than whole chunks).
"""

import math
import re
from collections.abc import Sequence
from typing import Protocol

_FENCE = re.compile(r"```.*?```", re.S)
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 3


class EntailmentScorer(Protocol):
    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


def split_sentences(text: str) -> list[str]:
    """Prose sentences only: drops code fences, tables, heading-like lines and short fragments."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", _FENCE.sub("\n\n", text)):
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not ln.startswith(("#", "|", "<", "{"))]
        para = " ".join(lines)
        if not para or not re.search(r"[.!?:]$", para):
            continue  # headings, labels and list stubs without terminal punctuation
        for sent in _SENT_END.split(para):
            if len(sent.split()) >= _MIN_WORDS:
                out.append(sent)
    return out


def premise_units(contexts: Sequence[str]) -> list[str]:
    """Each context, plus each of its sentences (>= 3 words), deduplicated in order."""
    units: list[str] = []
    for ctx in contexts:
        units.append(ctx)
        flat = " ".join(_FENCE.sub(" ", ctx).split())
        units.extend(s for s in _SENT_END.split(flat) if len(s.split()) >= _MIN_WORDS)
    return list(dict.fromkeys(units))


def faithfulness_score(
    answer: str, contexts: Sequence[str], judge: EntailmentScorer, threshold: float = 0.5
) -> dict:
    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return {"sentences": len(sentences), "supported": 0, "score": None}
    units = premise_units(contexts)
    pairs = [(u, s) for s in sentences for u in units]
    probs = judge.entailment(pairs)
    n_u = len(units)
    supported = sum(
        1 for i in range(len(sentences)) if max(probs[i * n_u : (i + 1) * n_u]) >= threshold
    )
    return {
        "sentences": len(sentences),
        "supported": supported,
        "score": round(supported / len(sentences), 4),
    }


class NLIJudge:
    """Local cross-encoder NLI model; label order is read from the model config."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, device="cpu")
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self._ent, self._con = labels["entailment"], labels["contradiction"]

    def scores(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        """(P(entailment), P(contradiction)) per (premise, hypothesis) pair."""
        if not pairs:
            return []
        logits = self.model.predict(list(pairs), batch_size=16, convert_to_numpy=True)
        out = []
        for row in logits:
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            out.append((exps[self._ent] / sum(exps), exps[self._con] / sum(exps)))
        return out

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [e for e, _ in self.scores(pairs)]


def evaluate_faithfulness(
    items: list[dict], judge: EntailmentScorer, threshold: float = 0.5
) -> dict:
    """Mean faithfulness over items {answer, contexts}; items with no scorable sentence are skipped.

    negative_control rotates answers so each is judged against another question's contexts; a
    judge that discriminates should score that much lower than the real pairing.
    """

    def mean(pairs: list[tuple[str, list[str]]]) -> tuple[int, float | None]:
        scores = [faithfulness_score(a, c, judge, threshold)["score"] for a, c in pairs]
        scores = [s for s in scores if s is not None]
        return len(scores), (round(sum(scores) / len(scores), 4) if scores else None)

    real = [(it["answer"], it["contexts"]) for it in items]
    n, score = mean(real)
    rotated = [(real[(i + 1) % len(real)][0], real[i][1]) for i in range(len(real))]
    _, neg = mean(rotated) if len(real) > 1 else (0, None)
    return {"n": n, "faithfulness": score, "negative_control": neg, "threshold": threshold}
