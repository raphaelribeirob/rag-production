"""Ingest a directory of Markdown files into the vector store (idempotent by chunk id)."""

from pathlib import Path

from rag_production.chunking import chunk_markdown
from rag_production.embeddings import Embedder
from rag_production.store import VectorStore


def ingest_dir(
    corpus_dir: str | Path,
    store: VectorStore,
    embedder: Embedder,
    max_chars: int = 1200,
    overlap: int = 0,
) -> dict[str, int]:
    docs = sorted(Path(corpus_dir).glob("*.md"))
    n_chunks = 0
    for doc in docs:
        chunks = chunk_markdown(doc.name, doc.read_text(encoding="utf-8"), max_chars, overlap)
        store.delete_doc(doc.name)  # drop stale chunks if the doc shrank
        if not chunks:
            continue
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert([(c.chunk_id, c.text, v) for c, v in zip(chunks, vectors, strict=True)])
        n_chunks += len(chunks)
    return {"documents": len(docs), "chunks": n_chunks}
