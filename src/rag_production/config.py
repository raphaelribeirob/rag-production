"""Runtime settings read from environment variables (see .env.example)."""

import hmac
import os
from dataclasses import dataclass, field

VALID_PROVIDERS = ("extractive", "mock", "ollama", "llamacpp", "hosted_free")


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)  # may embed a password (e.g. Neon)
    embedding_model: str
    llm_provider: str
    top_k: int
    api_key: str = field(default="", repr=False)
    corpus_dir: str = "data/corpus"
    abstain_threshold: float = 0.40  # top-1 cosine below this -> abstain
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    llamacpp_model_path: str = ""


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
        api_key=env.get("API_KEY", ""),
        corpus_dir=env.get("CORPUS_DIR", "data/corpus"),
        abstain_threshold=float(env.get("ABSTAIN_THRESHOLD", "0.40")),
        ollama_model=env.get("OLLAMA_MODEL", "qwen2.5:7b"),
        ollama_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        llamacpp_model_path=env.get("LLAMACPP_MODEL_PATH", ""),
    )


class IngestAuthError(Exception):
    """Raised when ingest is not authorized. status 503 = no key configured, 401 = bad key."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


def authorize_ingest(settings: Settings, provided: str | None) -> None:
    """Fail closed: ingest requires a configured API_KEY and a matching provided key."""
    if not settings.api_key:
        raise IngestAuthError(503, "ingest disabled: API_KEY not configured")
    if not provided or not hmac.compare_digest(provided, settings.api_key):
        raise IngestAuthError(401, "invalid or missing API key")
