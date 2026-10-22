"""Postgres + pgvector vector store (Neon-compatible). Configured via DATABASE_URL."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

_TABLE_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    score: float  # cosine similarity


class VectorStore:
    def __init__(self, database_url: str, dim: int = 384, table: str = "chunks") -> None:
        if not _TABLE_RE.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.dim = dim
        self.table = table
        self._t = sql.Identifier(table)
        self._idx = sql.Identifier(f"{table}_embedding_hnsw")
        self.conn = psycopg.connect(database_url, autocommit=True, connect_timeout=5)

    def migrate(self) -> None:
        """Create extension, table and HNSW cosine index. Safe to run repeatedly."""
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (id text PRIMARY KEY, text text NOT NULL, "
                "embedding vector({}) NOT NULL)"
            ).format(self._t, sql.Literal(self.dim))
        )
        self.conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)"
            ).format(self._idx, self._t)
        )

    def upsert(self, rows: Sequence[tuple[str, str, Sequence[float]]]) -> None:
        for _, _, emb in rows:
            if len(emb) != self.dim:
                raise ValueError(f"embedding dim {len(emb)} != {self.dim}")
        q = sql.SQL(
            "INSERT INTO {} (id, text, embedding) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding"
        ).format(self._t)
        with self.conn.cursor() as cur:
            cur.executemany(q, [(i, t, np.asarray(e, dtype=np.float32)) for i, t, e in rows])

    def search(self, query: Sequence[float], k: int = 5) -> list[Hit]:
        q = sql.SQL(
            "SELECT id, text, 1 - (embedding <=> %s) AS score FROM {} "
            "ORDER BY embedding <=> %s LIMIT %s"
        ).format(self._t)
        v = np.asarray(query, dtype=np.float32)
        rows = self.conn.execute(q, (v, v, k)).fetchall()
        return [Hit(r[0], r[1], float(r[2])) for r in rows]

    def count(self) -> int:
        return self.conn.execute(sql.SQL("SELECT count(*) FROM {}").format(self._t)).fetchone()[0]

    def has_extension(self) -> bool:
        r = self.conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        return r is not None

    def has_hnsw_index(self) -> bool:
        r = self.conn.execute(
            "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexdef ILIKE '%%hnsw%%'",
            (self.table,),
        ).fetchone()
        return r is not None

    def clear(self) -> None:
        self.conn.execute(sql.SQL("TRUNCATE {}").format(self._t))

    def drop(self) -> None:
        self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._t))

    def close(self) -> None:
        self.conn.close()
