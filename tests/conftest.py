import os

import pytest

DEFAULT_TEST_DB = "postgresql://rag:rag@localhost:5432/rag"


@pytest.fixture
def store():
    """VectorStore on a real Postgres+pgvector. CI sets REQUIRE_DB=1 so it never skips."""
    psycopg = pytest.importorskip("psycopg")
    from rag_production.store import VectorStore

    url = os.environ.get("DATABASE_URL", DEFAULT_TEST_DB)
    try:
        s = VectorStore(url, dim=3, table="chunks_test")
    except psycopg.OperationalError:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip("Postgres+pgvector not reachable (set DATABASE_URL)")
    s.migrate()
    s.clear()
    yield s
    s.drop()
    s.close()
