"""FastAPI app: /health, POST /ingest (API key), POST /query (top-k retrieval + cost/latency)."""

import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rag_production.config import IngestAuthError, Settings, authorize_ingest
from rag_production.embeddings import Embedder
from rag_production.ingest import ingest_dir
from rag_production.store import VectorStore

MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 20


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def create_app(settings: Settings, store: VectorStore, embedder: Embedder) -> FastAPI:
    app = FastAPI(title="rag-production")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": embedder.name}

    @app.post("/ingest")
    def ingest(x_api_key: str | None = Header(default=None)) -> dict:
        # The corpus path comes from server config only; clients cannot choose paths.
        try:
            authorize_ingest(settings, x_api_key)
        except IngestAuthError as e:
            raise HTTPException(status_code=e.status, detail=str(e)) from None
        return ingest_dir(settings.corpus_dir, store, embedder)

    @app.post("/query")
    def query(body: QueryIn) -> dict:
        t0 = time.perf_counter()
        (vec,) = embedder.embed([body.question])
        embed_ms = _ms(t0)
        t1 = time.perf_counter()
        hits = store.search(vec, k=body.top_k)
        retrieve_ms = _ms(t1)
        citations = [
            {"chunk_id": h.chunk_id, "doc_id": h.chunk_id.split("#")[0], "score": h.score}
            for h in hits
        ]
        answer = hits[0].text if hits else ""
        return {
            "answer": answer,  # extractive provider: best chunk verbatim, no LLM call
            "citations": citations,
            "provider": "extractive",
            "cost_usd": 0.0,  # local embeddings + extractive answer = $0
            "latency_ms": {"embed": embed_ms, "retrieve": retrieve_ms, "total": _ms(t0)},
            "request_id": str(uuid.uuid4()),
        }

    return app


def build_default_app() -> FastAPI:  # pragma: no cover - wiring for `uvicorn --factory`
    from rag_production.config import load_settings
    from rag_production.embeddings import SentenceTransformerEmbedder

    settings = load_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.database_url, dim=embedder.dim)
    store.migrate()
    return create_app(settings, store, embedder)
