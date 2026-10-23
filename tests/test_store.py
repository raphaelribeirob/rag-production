import pytest


def test_migrate_creates_extension_and_hnsw_index(store):
    assert store.has_extension()
    assert store.has_hnsw_index()


def test_migrate_is_idempotent(store):
    store.migrate()
    assert store.has_hnsw_index()


def test_upsert_and_cosine_top_k(store):
    store.upsert(
        [
            ("a", "doc a", [1.0, 0.0, 0.0]),
            ("b", "doc b", [0.0, 1.0, 0.0]),
            ("c", "doc c", [0.9, 0.1, 0.0]),
        ]
    )
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert [h.chunk_id for h in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(1.0)


def test_upsert_same_id_is_idempotent(store):
    store.upsert([("a", "v1", [1.0, 0.0, 0.0])])
    store.upsert([("a", "v2", [0.0, 1.0, 0.0])])
    assert store.count() == 1
    assert store.search([0.0, 1.0, 0.0], k=1)[0].text == "v2"


def test_wrong_dimension_rejected(store):
    with pytest.raises(ValueError):
        store.upsert([("x", "bad", [1.0, 0.0])])


def test_invalid_table_name_rejected():
    from rag_production.store import VectorStore

    with pytest.raises(ValueError):
        VectorStore("postgresql://unused", dim=3, table="chunks; DROP TABLE x")


def test_sql_injection_payload_is_stored_as_data(store):
    payload = "x'); DROP TABLE chunks_test; --"
    store.upsert([(payload, payload, [1.0, 0.0, 0.0])])
    hit = store.search([1.0, 0.0, 0.0], k=1)[0]
    assert (hit.chunk_id, hit.text) == (payload, payload)
    assert store.count() == 1


def test_fresh_instance_can_search_and_upsert_without_migrate(store):
    """Regression: vector type was only registered in migrate(), so a second instance failed."""
    import os

    from rag_production.store import VectorStore

    store.upsert([("a", "doc a", [1.0, 0.0, 0.0])])
    url = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    other = VectorStore(url, dim=3, table="chunks_test")  # no migrate() on this instance
    try:
        other.upsert([("b", "doc b", [0.0, 1.0, 0.0])])
        assert [h.chunk_id for h in other.search([1.0, 0.0, 0.0], k=1)] == ["a"]
    finally:
        other.close()
