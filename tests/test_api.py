import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag_production.api import create_app  # noqa: E402
from rag_production.config import load_settings  # noqa: E402
from rag_production.embeddings import HashEmbedder  # noqa: E402

KEY = "test-key-not-a-secret"


@pytest.fixture
def client(store, tmp_path):
    (tmp_path / "cors.md").write_text("# CORS\n\nUse CORSMiddleware to allow origins.\n")
    (tmp_path / "files.md").write_text("# Files\n\nUpload files with UploadFile.\n")
    settings = load_settings({"API_KEY": KEY, "CORPUS_DIR": str(tmp_path)})
    return TestClient(create_app(settings, store, HashEmbedder(dim=3)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ingest_requires_key(client):
    assert client.post("/ingest").status_code == 401
    assert client.post("/ingest", headers={"X-API-Key": "nope"}).status_code == 401


def test_ingest_disabled_without_configured_key(store):
    app = create_app(load_settings({}), store, HashEmbedder(dim=3))
    assert TestClient(app).post("/ingest", headers={"X-API-Key": "x"}).status_code == 503


def test_ingest_then_query_returns_citations_cost_latency(client):
    r = client.post("/ingest", headers={"X-API-Key": KEY})
    assert r.status_code == 200 and r.json() == {"documents": 2, "chunks": 2}
    r = client.post("/query", json={"question": "upload files", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["citations"]) == 2
    assert {"chunk_id", "doc_id", "score"} <= set(body["citations"][0])
    assert body["cost_usd"] == 0.0
    assert body["provider"] == "extractive"
    assert {"embed", "retrieve", "total"} <= set(body["latency_ms"])
    assert body["latency_ms"]["total"] >= 0


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "x" * 2001}, {"question": "ok", "top_k": 21}, {"top_k": 5}],
)
def test_query_input_limits(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_ingest_does_not_accept_client_supplied_paths(client):
    r = client.post("/ingest", headers={"X-API-Key": KEY}, json={"path": "/etc"})
    assert r.status_code == 200 and r.json()["documents"] == 2


def test_query_abstains_below_threshold(store, tmp_path):
    (tmp_path / "cors.md").write_text("# CORS\n\nUse CORSMiddleware.\n")
    settings = load_settings(
        {"API_KEY": KEY, "CORPUS_DIR": str(tmp_path), "ABSTAIN_THRESHOLD": "1.01"}
    )
    c = TestClient(create_app(settings, store, HashEmbedder(dim=3)))
    c.post("/ingest", headers={"X-API-Key": KEY})
    body = c.post("/query", json={"question": "CORS"}).json()
    assert body["abstained"] is True
    assert body["answer"] == ""
    assert body["citations"]  # still returned for transparency


def test_query_answers_above_threshold(client):
    client.post("/ingest", headers={"X-API-Key": KEY})
    body = client.post("/query", json={"question": "upload files"}).json()
    assert body["abstained"] is False and body["answer"]
