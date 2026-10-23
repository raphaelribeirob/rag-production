from rag_production.embeddings import HashEmbedder
from rag_production.ingest import ingest_dir


def _corpus(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\nCORS middleware settings.\n\n## Beta\n\nMore.\n")
    (tmp_path / "b.md").write_text("# Files\n\nUpload files with UploadFile.\n")
    (tmp_path / "LICENSE").write_text("MIT")
    return tmp_path


def test_ingest_dir_loads_markdown_into_store(store, tmp_path):
    stats = ingest_dir(_corpus(tmp_path), store, HashEmbedder(dim=3))
    assert stats == {"documents": 2, "chunks": 3}
    assert store.count() == 3


def test_ingest_is_idempotent(store, tmp_path):
    corpus = _corpus(tmp_path)
    ingest_dir(corpus, store, HashEmbedder(dim=3))
    ingest_dir(corpus, store, HashEmbedder(dim=3))
    assert store.count() == 3


def test_ingest_uses_chunk_size_and_overlap(store, tmp_path):
    body = "\n\n".join(f"para{i} " + "word " * 40 for i in range(6))
    (tmp_path / "long.md").write_text("# Long\n\n" + body + "\n")
    small = ingest_dir(tmp_path, store, HashEmbedder(dim=3), max_chars=300, overlap=50)
    store.clear()
    default = ingest_dir(tmp_path, store, HashEmbedder(dim=3))
    assert small["chunks"] > default["chunks"] and store.count() == default["chunks"]
