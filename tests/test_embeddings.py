import math

import pytest

from rag_production.embeddings import HashEmbedder


def test_hash_embedder_is_deterministic_normalized_384():
    e = HashEmbedder()
    v1, v2 = e.embed(["path parameters", "path parameters"])
    assert e.dim == len(v1) == 384
    assert v1 == v2
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-6)


def test_sentence_transformer_embedder_dim_384():
    pytest.importorskip("sentence_transformers")
    from rag_production.embeddings import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    (v,) = e.embed(["How do I declare a path parameter?"])
    assert e.dim == len(v) == 384
