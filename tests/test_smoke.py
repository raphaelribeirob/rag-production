import pytest

from rag_production import __version__
from rag_production.config import load_settings
from rag_production.metrics import hit_at_k, recall_at_k


def test_package_imports():
    assert __version__


def test_default_settings_use_no_model_fallback():
    s = load_settings({})
    assert s.llm_provider == "extractive"
    assert s.top_k == 5


def test_invalid_provider_rejected():
    with pytest.raises(ValueError):
        load_settings({"LLM_PROVIDER": "paid_api"})


def test_metrics_on_toy_example():
    retrieved = ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert hit_at_k(retrieved, {"c3"}) == 1.0
    assert hit_at_k(retrieved, {"c6"}) == 0.0
    assert recall_at_k(retrieved, {"c1", "c6"}) == 0.5
