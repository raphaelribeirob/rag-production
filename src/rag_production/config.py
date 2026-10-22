"""Runtime settings read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass

VALID_PROVIDERS = ("extractive", "ollama", "hosted_free")


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    llm_provider: str
    top_k: int


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ) if env is None else env
    provider = env.get("LLM_PROVIDER", "extractive")
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {provider!r}")
    return Settings(
        database_url=env.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"),
        embedding_model=env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        llm_provider=provider,
        top_k=int(env.get("TOP_K", "5")),
    )
