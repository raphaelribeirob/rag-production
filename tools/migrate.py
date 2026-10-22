"""Create pgvector extension, chunks table and HNSW index.

Usage: DATABASE_URL=... python tools/migrate.py
"""

from rag_production.config import load_settings
from rag_production.store import VectorStore

if __name__ == "__main__":
    store = VectorStore(load_settings().database_url)
    store.migrate()
    store.close()
    print("migration ok")
