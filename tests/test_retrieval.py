"""Retriever: dense top-k, or dense top-N candidates reranked by a cross-encoder to top-k."""

import pytest

from rag_production.config import load_settings
from rag_production.retrieval import Retriever, table_for
from rag_production.store import Hit

HITS = [Hit(f"d{i}.md#0", "x" * (i + 1), 0.9 - i * 0.05) for i in range(10)]


class FakeStore:
    def __init__(self):
        self.ks = []

    def search(self, vec, k=5):
        self.ks.append(k)
        return HITS[:k]


class FakeEmbedder:
    name, dim = "fake", 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class LengthReranker:
    name = "fake-reranker"

    def score(self, question, texts):
        return [float(len(t)) for t in texts]  # longest text ranks first


def test_dense_only_keeps_cosine_order_and_fetches_k():
    store = FakeStore()
    r = Retriever(store, FakeEmbedder()).retrieve("q", k=5)
    assert store.ks == [5]
    assert [h.chunk_id for h in r.hits] == [h.chunk_id for h in HITS[:5]]
    assert r.top_cosine == 0.9 and r.rerank_scores is None
    assert set(r.ms) == {"embed", "retrieve"}


def test_reranker_reorders_candidates_and_keeps_abstention_on_dense_cosine():
    store = FakeStore()
    r = Retriever(store, FakeEmbedder(), LengthReranker(), candidates=8).retrieve("q", k=5)
    assert store.ks == [8]
    assert [h.chunk_id for h in r.hits] == [f"d{i}.md#0" for i in (7, 6, 5, 4, 3)]
    assert r.rerank_scores == [8.0, 7.0, 6.0, 5.0, 4.0]
    assert r.top_cosine == 0.9  # max dense cosine, so the 0.40 threshold means the same thing
    assert set(r.ms) == {"embed", "retrieve", "rerank"}


def test_ranked_for_retrieval_eval_reranks_all_candidates():
    store = FakeStore()
    ranked = Retriever(store, FakeEmbedder(), LengthReranker(), candidates=4).ranked("q", 6)
    assert store.ks == [6]  # at least n candidates
    assert [cid for cid, _ in ranked] == [f"d{i}.md#0" for i in (5, 4, 3, 2, 1, 0)]
    assert max(s for _, s in ranked) == 0.9  # cosine kept per chunk


def test_table_per_chunk_config_keeps_the_baseline_table():
    assert table_for(load_settings({}), "chunks_eval") == "chunks_eval"
    env = {"CHUNK_SIZE": "500", "CHUNK_OVERLAP": "100"}
    assert table_for(load_settings(env), "chunks_eval") == "chunks_eval_c500_o100"


def test_retrieval_settings_defaults_and_validation():
    s = load_settings({})
    assert (s.chunk_size, s.chunk_overlap, s.reranker, s.rerank_candidates) == (1200, 0, "", 20)
    assert s.reranker_revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    s = load_settings(
        {"RERANKER": "cross-encoder/ms-marco-MiniLM-L6-v2", "RERANK_CANDIDATES": "30"}
    )
    assert s.reranker.endswith("MiniLM-L6-v2") and s.rerank_candidates == 30
    for bad in ({"CHUNK_SIZE": "100"}, {"CHUNK_SIZE": "400", "CHUNK_OVERLAP": "400"},
                {"CHUNK_OVERLAP": "-1"}, {"RERANK_CANDIDATES": "0"}):  # fmt: skip
        with pytest.raises(ValueError):
            load_settings(bad)


def test_get_reranker_is_off_by_default():
    from rag_production.retrieval import get_reranker

    assert get_reranker(load_settings({})) is None


def test_reranker_model_pinned_to_full_hf_revision():
    import re

    from rag_production.config import RERANKER_REVISION, load_settings

    assert re.fullmatch(r"[0-9a-f]{40}", RERANKER_REVISION)
    s = load_settings({"RERANKER": "cross-encoder/ms-marco-MiniLM-L6-v2"})
    assert s.reranker_revision == RERANKER_REVISION
