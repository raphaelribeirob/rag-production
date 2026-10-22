"""Retrieval metrics used by the eval gate (SPEC: hit rate@5, recall@5)."""

from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """1.0 if any relevant id appears in the top-k retrieved ids, else 0.0."""
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """Fraction of relevant ids found in the top-k retrieved ids."""
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


# TODO: faithfulness scorer (LLM-judge or NLI-based) - not built yet.
