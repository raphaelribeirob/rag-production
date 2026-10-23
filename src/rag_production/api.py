"""FastAPI app: /health, POST /ingest (API key), POST /query (retrieval, answer, cost/latency)."""

import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rag_production.config import IngestAuthError, Settings, authorize_ingest
from rag_production.embeddings import Embedder
from rag_production.ingest import ingest_dir
from rag_production.providers import LLMProvider, get_provider
from rag_production.retrieval import Reranker, Retriever, get_reranker, table_for
from rag_production.store import VectorStore

MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 20


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def create_app(
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    provider: LLMProvider | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    app = FastAPI(title="rag-production")
    retriever = Retriever(store, embedder, reranker, settings.rerank_candidates)

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
        return ingest_dir(
            settings.corpus_dir, store, embedder, settings.chunk_size, settings.chunk_overlap
        )

    @app.post("/query")
    def query(body: QueryIn) -> dict:
        t0 = time.perf_counter()
        found = retriever.retrieve(body.question, k=body.top_k)
        hits = found.hits
        citations = [
            {"chunk_id": h.chunk_id, "doc_id": h.chunk_id.split("#")[0], "score": h.score}
            for h in hits
        ]
        for c, s in zip(citations, found.rerank_scores or [], strict=False):
            c["rerank_score"] = s  # cross-encoder logit; "score" stays the cosine
        abstained = not hits or found.top_cosine < settings.abstain_threshold
        t2 = time.perf_counter()
        if abstained:
            answer = ""
        elif provider is None:
            answer = hits[0].text  # extractive: best chunk verbatim, no LLM call
        else:
            answer = provider.generate(body.question, [h.text for h in hits])
        latency = dict(found.ms)
        if provider is not None:
            latency["generate"] = _ms(t2)
        latency["total"] = _ms(t0)
        return {
            "answer": answer,
            "abstained": abstained,
            "citations": citations,
            "provider": provider.name if provider else "extractive",
            "answer_source": "generated" if provider else "extractive",
            "cost_usd": provider.cost_usd if provider else 0.0,  # local only: $0
            "latency_ms": latency,
            "request_id": str(uuid.uuid4()),
        }

    return app


def build_default_app() -> FastAPI:  # pragma: no cover - wiring for `uvicorn --factory`
    from rag_production.config import load_settings
    from rag_production.embeddings import SentenceTransformerEmbedder

    settings = load_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(
        settings.database_url, dim=embedder.dim, table=table_for(settings, "chunks")
    )
    store.migrate()
    provider, reranker = get_provider(settings), get_reranker(settings)
    return create_app(settings, store, embedder, provider=provider, reranker=reranker)
