"""LLM providers: zero spend, no accounts. `None` provider = extractive answer (no model call).

- MockProvider: deterministic, used in CI. Copies the first full prose sentence of the top 2 chunks.
  It tests the generation code path, not answer quality.
- OllamaProvider: local Ollama (default qwen2.5:7b), optional, never used in CI. Localhost only.
- LlamaCppProvider: in-process llama.cpp on a local GGUF file, optional, never used in CI.
"""

import json
import os
import re
import urllib.request
from collections.abc import Callable, Sequence
from typing import Protocol
from urllib.parse import urlparse

from rag_production.faithfulness import split_sentences

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROMPT = (
    "Answer the question using only the context below. If the context does not contain the "
    "answer, say you don't know. Answer in at most 3 plain sentences, no code.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class LLMProvider(Protocol):
    name: str
    cost_usd: float

    def generate(self, question: str, contexts: Sequence[str]) -> str: ...


class MockProvider:
    name = "mock"
    cost_usd = 0.0

    def __init__(self, max_contexts: int = 2) -> None:
        self.max_contexts = max_contexts

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        picked = []
        for ctx in contexts[: self.max_contexts]:
            # skip lead-ins ending in ":" (they introduce code); joined, they would merge sentences
            sentences = [s for s in split_sentences(ctx) if s[-1] in ".!?"]
            if sentences:
                picked.append(sentences[0])
        return " ".join(picked)


def _http_post(url: str, payload: dict, timeout: float) -> dict:  # pragma: no cover - network
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost only
        return json.loads(resp.read())


class OllamaProvider:
    cost_usd = 0.0

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        post: Callable[[str, dict, float], dict] = _http_post,
    ) -> None:
        u = urlparse(base_url)
        if u.scheme != "http" or u.hostname not in LOCAL_HOSTS:
            raise ValueError("OllamaProvider only talks to a local Ollama (http://localhost)")
        self.model, self.base_url, self.timeout, self._post = model, base_url, timeout, post
        self.name = f"ollama:{model}"

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        context = "\n\n---\n\n".join(contexts)
        payload = {
            "model": self.model,
            "prompt": PROMPT.format(context=context, question=question),
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }
        out = self._post(f"{self.base_url.rstrip('/')}/api/generate", payload, self.timeout)
        return re.sub(r"\s+", " ", out.get("response", "")).strip()


class LlamaCppProvider:
    """In-process generation (no server, no network) with llama-cpp-python (optional extra)."""

    cost_usd = 0.0

    def __init__(self, model_path: str, n_ctx: int = 4096, max_tokens: int = 200, llm=None):
        base = re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(model_path))
        self.name = f"llamacpp:{base}"
        self.max_tokens = max_tokens
        if llm is None:  # pragma: no cover - needs the optional package and a model file
            from llama_cpp import Llama

            llm = Llama(model_path=model_path, n_ctx=n_ctx, seed=0, verbose=False)
        self._llm = llm

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        prompt = PROMPT.format(context="\n\n---\n\n".join(contexts), question=question)
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=0,
            max_tokens=self.max_tokens,
        )
        return re.sub(r"\s+", " ", out["choices"][0]["message"]["content"] or "").strip()


def get_provider(settings, llm=None) -> LLMProvider | None:
    if settings.llm_provider == "extractive":
        return None
    if settings.llm_provider == "mock":
        return MockProvider()
    if settings.llm_provider == "ollama":
        return OllamaProvider(model=settings.ollama_model, base_url=settings.ollama_url)
    if settings.llm_provider == "llamacpp":
        if not settings.llamacpp_model_path:
            raise ValueError("LLM_PROVIDER=llamacpp needs LLAMACPP_MODEL_PATH (a local .gguf file)")
        return LlamaCppProvider(settings.llamacpp_model_path, llm=llm)
    raise NotImplementedError(f"provider {settings.llm_provider!r} is not implemented")
