"""Runtime settings read from environment variables (see .env.example)."""

import hmac
import os
from dataclasses import dataclass, field

from rag_production.prompts import PROMPTS

# Hugging Face commit of cross-encoder/ms-marco-MiniLM-L6-v2 (model.safetensors sha256 821d1aa6...)
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
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
    llm_prompt: str = "v1"
    llm_temperature: float = 0.0  # 0 = greedy (the baseline); >0 samples with llm_seed
    llm_seed: int = 0
    chunk_size: int = 1200  # max chars per chunk (heading included); 1200 = baseline chunker
    chunk_overlap: int = 0
    reranker: str = ""  # "" = off (baseline); e.g. cross-encoder/ms-marco-MiniLM-L6-v2
    reranker_revision: str = RERANKER_REVISION
    rerank_candidates: int = 20  # dense candidates reranked down to top_k


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ) if env is None else env
    provider = env.get("LLM_PROVIDER", "extractive")
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {provider!r}")
    prompt = env.get("LLM_PROMPT", "v1")
    if prompt not in PROMPTS:
        raise ValueError(f"LLM_PROMPT must be one of {sorted(PROMPTS)}, got {prompt!r}")
    chunk_size, overlap = int(env.get("CHUNK_SIZE", "1200")), int(env.get("CHUNK_OVERLAP", "0"))
    if chunk_size < 200 or not 0 <= overlap < chunk_size // 2:
        raise ValueError("need CHUNK_SIZE >= 200 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE / 2")
    candidates = int(env.get("RERANK_CANDIDATES", "20"))
    if candidates < 1:
        raise ValueError("RERANK_CANDIDATES must be >= 1")
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
        llm_prompt=prompt,
        llm_temperature=float(env.get("LLM_TEMPERATURE", "0")),
        llm_seed=int(env.get("LLM_SEED", "0")),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        reranker=env.get("RERANKER", ""),
        reranker_revision=env.get("RERANKER_REVISION", RERANKER_REVISION),
        rerank_candidates=candidates,
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
