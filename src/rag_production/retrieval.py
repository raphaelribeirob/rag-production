"""Retrieval: dense pgvector top-k, optionally dense top-N candidates reranked to top-k.

Abstention always uses the best dense cosine among the candidates, so ABSTAIN_THRESHOLD (0.40)
keeps its meaning with or without the reranker.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_production.store import Hit


class Reranker(Protocol):
    name: str

    def score(self, question: str, texts: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True)
class Retrieved:
    hits: list[Hit]  # final top-k, in the order given to the LLM
    top_cosine: float  # best dense cosine among candidates (abstention input)
    rerank_scores: list[float] | None
    ms: dict[str, float]  # embed, retrieve and (when reranking) rerank


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


class Retriever:
    def __init__(self, store, embedder, reranker: Reranker | None = None, candidates: int = 20):
        self.store, self.embedder, self.reranker, self.candidates = (
            store,
            embedder,
            reranker,
            candidates,
        )

    def _search(self, question: str, k: int) -> tuple[list[Hit], dict[str, float]]:
        t0 = time.perf_counter()
        (vec,) = self.embedder.embed([question])
        ms = {"embed": _ms(t0)}
        t1 = time.perf_counter()
        hits = self.store.search(vec, k=k)
        ms["retrieve"] = _ms(t1)
        return hits, ms

    def _rerank(self, question: str, hits: list[Hit], ms: dict) -> list[tuple[Hit, float]]:
        t0 = time.perf_counter()
        scores = self.reranker.score(question, [h.text for h in hits]) if hits else []
        ms["rerank"] = _ms(t0)
        order = sorted(range(len(hits)), key=lambda i: -scores[i])  # stable on ties
        return [(hits[i], float(scores[i])) for i in order]

    def retrieve(self, question: str, k: int = 5) -> Retrieved:
        if self.reranker is None:
            hits, ms = self._search(question, k)
            return Retrieved(hits, hits[0].score if hits else 0.0, None, ms)
        hits, ms = self._search(question, max(k, self.candidates))
        top = max((h.score for h in hits), default=0.0)
        ranked = self._rerank(question, hits, ms)[:k]
        return Retrieved([h for h, _ in ranked], top, [s for _, s in ranked], ms)

    def ranked(self, question: str, n: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine) for the retrieval eval: n chunks, reranked when a reranker is set."""
        if self.reranker is None:
            hits, _ = self._search(question, n)
            return [(h.chunk_id, h.score) for h in hits]
        hits, ms = self._search(question, max(n, self.candidates))
        return [(h.chunk_id, h.score) for h, _ in self._rerank(question, hits, ms)[:n]]


class CrossEncoderReranker:  # pragma: no cover - needs the model download (not in CI)
    """Local sentence-transformers cross-encoder pinned to a Hugging Face revision (CPU, $0)."""

    def __init__(self, model_name: str, revision: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"{model_name}@{revision[:7]}"
        self.model = CrossEncoder(model_name, revision=revision, device="cpu")

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        pairs = [(question, t) for t in texts]
        return [float(x) for x in self.model.predict(pairs, batch_size=32)]


def get_reranker(settings) -> Reranker | None:
    if not settings.reranker:
        return None
    return CrossEncoderReranker(settings.reranker, settings.reranker_revision)  # pragma: no cover


def table_for(settings, base: str) -> str:
    """Baseline chunking keeps `base`; other chunk configs get their own table."""
    if (settings.chunk_size, settings.chunk_overlap) == (1200, 0):
        return base
    return f"{base}_c{settings.chunk_size}_o{settings.chunk_overlap}"
