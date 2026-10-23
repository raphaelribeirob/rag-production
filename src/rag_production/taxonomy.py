"""Failure taxonomy for generated answers. Rules fixed 2026-10-22 before seeing classifier output.

Per prose sentence (same `split_sentences` as the faithfulness judge), first match wins:
1. supported          max P(entailment) over premise units >= 0.5 (the CI judge's own rule)
2. evasive            refusal phrase ("the context does not ...") and a gold page is in the top-5
   justified_refusal  refusal phrase and no gold page in the top-5
3. hallucination      P(contradiction) >= 0.5, from the premise unit with the highest entailment
4. hallucination      an identifier (`code`, dotted.name or snake_case) not verbatim in any context
5. paraphrase         content-token recall >= 0.75 against the best single context chunk
6. hallucination      otherwise
Answer label = worst sentence: hallucination > evasive > paraphrase > justified_refusal >
supported. No scorable sentence = "unscored" (excluded from rates, as in the faithfulness score).
"""

import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from rag_production.faithfulness import premise_units, split_sentences

ENTAIL_T, CONTRA_T, OVERLAP_T = 0.5, 0.5, 0.75
LABELS = ("supported", "hallucination", "evasive", "paraphrase", "justified_refusal")
_PRIORITY = ("hallucination", "evasive", "paraphrase", "justified_refusal", "supported")
_DOC = r"(context|text|documentation|docs|passage|information provided|provided information)"
_REFUSAL = [
    re.compile(
        rf"\b{_DOC}\b[^.]{{0,60}}?\b(does not|doesn't|do not|don't|did not|didn't|is not|isn't|"
        r"lacks)\b",
        re.I,
    ),
    re.compile(
        r"\b(not|isn't|aren't|wasn't) (explicitly )?(mentioned|specified|provided|detailed|covered|"
        rf"included|described|explained|stated|given)\b[^.]{{0,40}}\b{_DOC}\b",
        re.I,
    ),
    re.compile(
        r"\bI (don't|do not) know\b|\b(cannot|can't|unable to) (answer|determine|tell)\b"
        r"|\bno (information|details|mention)\b",
        re.I,
    ),
]
_TICK = re.compile(r"`([^`\s]+)`")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
_STOP = set(
    "the and for are you can with that this use your from not but has have will into its it's "
    "also any all one more when then than their there them they what which who how why where "
    "like such each other some only just should would could may might must via using used "
    "does did was were been being about over under same both".split()
)


class NLIScorer(Protocol):
    def scores(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]: ...


def is_refusal(sentence: str) -> bool:
    return any(p.search(sentence) for p in _REFUSAL)


def novel_code_tokens(sentence: str, contexts: Sequence[str]) -> list[str]:
    """Identifiers in the sentence that appear verbatim in no context (code-like spans only)."""
    joined = "\n".join(contexts)
    found: list[str] = []
    rest = sentence
    for rx in (_TICK, _DOTTED, _SNAKE):
        for m in rx.finditer(rest):
            tok = (m.group(1) if m.groups() else m.group(0)).removesuffix("()").rstrip(".,:;")
            if re.fullmatch(r"[\w.]+", tok) and ("_" in tok or "." in tok or rx is _TICK):
                found.append(tok)
        rest = rx.sub(" ", rest)
    return [t for t in dict.fromkeys(found) if t not in joined]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) >= 3} - _STOP


def lexical_overlap(sentence: str, contexts: Sequence[str]) -> float:
    s = _tokens(sentence)
    if not s or not contexts:
        return 0.0
    return round(max(len(s & _tokens(c)) / len(s) for c in contexts), 4)


def classify_sentence(entail, contra, overlap, novel, refusal, gold_retrieved) -> str:
    if entail >= ENTAIL_T:
        return "supported"
    if refusal:
        return "evasive" if gold_retrieved else "justified_refusal"
    if contra >= CONTRA_T or novel:
        return "hallucination"
    return "paraphrase" if overlap >= OVERLAP_T else "hallucination"


def classify_answer(
    answer: str, contexts: Sequence[str], gold_retrieved: bool, judge: NLIScorer
) -> dict:
    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return {"label": "unscored", "score": None, "sentences": []}
    units = premise_units(contexts)
    probs = judge.scores([(u, s) for s in sentences for u in units])
    rows = []
    for i, sent in enumerate(sentences):
        block = probs[i * len(units) : (i + 1) * len(units)]
        ent, con = max(block, key=lambda ec: ec[0])
        novel = novel_code_tokens(sent, contexts)
        over, ref = lexical_overlap(sent, contexts), is_refusal(sent)
        label = classify_sentence(ent, con, over, novel, ref, gold_retrieved)
        rows.append(
            {"text": sent, "label": label, "entail": round(ent, 4), "contra": round(con, 4)}
            | {"overlap": over, "novel_code": novel, "refusal": ref}
        )
    labels = {r["label"] for r in rows}
    label = next(p for p in _PRIORITY if p in labels)
    supported = sum(r["label"] == "supported" for r in rows)
    return {"label": label, "score": round(supported / len(rows), 4), "sentences": rows}


def summarize(labels: Sequence[str]) -> dict:
    counts = Counter(labels)
    n = sum(counts[k] for k in LABELS)
    rates = {k: (round(counts[k] / n, 4) if n else None) for k in LABELS}
    return {"n": n, "unscored": counts["unscored"], "counts": {k: counts[k] for k in LABELS}} | {
        "rates": rates
    }


def agreement(hand: dict[str, str], auto: dict[str, str]) -> dict:
    keys = [k for k in hand if k in auto]
    agree = sum(hand[k] == auto[k] for k in keys)
    confusion = Counter(f"{hand[k]}->{auto[k]}" for k in keys if hand[k] != auto[k])
    rate = round(agree / len(keys), 4) if keys else None
    return {"n": len(keys), "agree": agree, "rate": rate, "confusion": dict(confusion)}
