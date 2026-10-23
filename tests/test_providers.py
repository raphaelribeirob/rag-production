import json

import pytest

from rag_production.config import load_settings
from rag_production.providers import MockProvider, OllamaProvider, get_provider

CTX = [
    "Path params\n\nYou can declare path parameters. They are passed to your function.",
    "Query params\n\nOther function parameters are query parameters.",
]


def test_mock_provider_is_deterministic_and_grounded():
    p = MockProvider()
    a1, a2 = p.generate("q?", CTX), p.generate("q?", CTX)
    assert a1 == a2 and a1
    for sentence in a1.split(". "):
        assert sentence.rstrip(".") in " ".join(CTX)
    assert p.name == "mock" and p.cost_usd == 0.0


def test_mock_provider_without_context_returns_empty():
    assert MockProvider().generate("q?", []) == ""


def test_get_provider_by_setting():
    assert get_provider(load_settings({})) is None  # extractive default: no model call
    assert isinstance(get_provider(load_settings({"LLM_PROVIDER": "mock"})), MockProvider)
    p = get_provider(load_settings({"LLM_PROVIDER": "ollama"}))
    assert isinstance(p, OllamaProvider) and p.model == "qwen2.5:7b"
    with pytest.raises(NotImplementedError):
        get_provider(load_settings({"LLM_PROVIDER": "hosted_free"}))


@pytest.mark.parametrize("url", ["https://api.example.com", "http://10.0.0.5:11434", "file:///x"])
def test_ollama_only_talks_to_localhost(url):
    with pytest.raises(ValueError):
        OllamaProvider(base_url=url)


def test_ollama_request_is_local_deterministic_and_context_only():
    seen = {}

    def fake_post(url, payload, timeout):
        seen.update(url=url, payload=payload, timeout=timeout)
        return {"response": " Path parameters are passed to your function. "}

    p = OllamaProvider(post=fake_post)
    out = p.generate("What are path params?", CTX)
    assert out == "Path parameters are passed to your function."
    assert seen["url"] == "http://localhost:11434/api/generate"
    body = seen["payload"]
    assert body["model"] == "qwen2.5:7b" and body["stream"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 0
    assert "only" in body["prompt"].lower() and CTX[0] in body["prompt"]
    assert seen["timeout"] > 0
    json.dumps(body)  # serializable


def test_mock_skips_colon_lead_ins_so_joined_sentences_stay_separate():
    ctx = ["H\n\nYou can add a parameter:\n\n```py\nx = 1\n```\n\nThe parameter is optional."]
    assert MockProvider().generate("q?", ctx) == "The parameter is optional."


def test_ollama_settings_from_env_still_localhost_only():
    env = {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://203.0.113.9:11434"}
    with pytest.raises(ValueError):
        get_provider(load_settings(env))
    env = {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "llama3.2:3b"}
    assert get_provider(load_settings(env)).name == "ollama:llama3.2:3b"


class FakeLlama:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kw):
        self.calls.append(kw)
        return {"choices": [{"message": {"content": "  Path params go\n in the URL.  "}}]}


def test_llamacpp_in_process_deterministic_context_only():
    from rag_production.providers import LlamaCppProvider

    fake = FakeLlama()
    p = LlamaCppProvider("/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", llm=fake)
    assert p.generate("What are path params?", CTX) == "Path params go in the URL."
    kw = fake.calls[0]
    assert kw["temperature"] == 0 and kw["seed"] == 0 and kw["max_tokens"] <= 256
    prompt = kw["messages"][-1]["content"]
    assert "only" in prompt.lower() and CTX[1] in prompt
    assert p.name == "llamacpp:qwen2.5-7b-instruct-q4_k_m" and p.cost_usd == 0.0


def test_llamacpp_selected_by_settings_and_requires_model_path():
    from rag_production.providers import LlamaCppProvider

    with pytest.raises(ValueError):
        get_provider(load_settings({"LLM_PROVIDER": "llamacpp"}))
    env = {"LLM_PROVIDER": "llamacpp", "LLAMACPP_MODEL_PATH": "/m/x.gguf"}
    p = get_provider(load_settings(env), llm=FakeLlama())
    assert isinstance(p, LlamaCppProvider)


def test_prompt_versions_baseline_frozen_and_selectable():
    from rag_production.prompts import PROMPTS

    # v1 is the 2026-10-22 baseline (faithfulness 0.647); it must never change
    assert PROMPTS["v1"].startswith("Answer the question using only the context below. If the")
    assert "say you don't know. Answer in at most 3 plain sentences, no code." in PROMPTS["v1"]
    with pytest.raises(ValueError):
        load_settings({"LLM_PROMPT": "nope"})
    assert load_settings({}).llm_prompt == "v1"
    fake = FakeLlama()
    last = sorted(PROMPTS)[-1]
    env = {"LLM_PROVIDER": "llamacpp", "LLAMACPP_MODEL_PATH": "/m/x.gguf", "LLM_PROMPT": last}
    p = get_provider(load_settings(env), llm=fake)
    p.generate("Q?", CTX)
    sent = fake.calls[0]["messages"][-1]["content"]
    assert sent.startswith(PROMPTS[last].split("{context}")[0]) and CTX[1] in sent
    assert p.prompt_version == last
