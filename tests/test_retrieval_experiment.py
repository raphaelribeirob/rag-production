import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT / "tools" / "retrieval_experiment.py"
    spec = importlib.util.spec_from_file_location("rx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_config_is_the_default_and_each_chunk_config_has_its_own_table():
    from rag_production.config import load_settings
    from rag_production.retrieval import table_for

    tool = _tool()
    base = load_settings(tool.CONFIGS["c1200"])
    default = load_settings({})
    assert (base.chunk_size, base.chunk_overlap, base.reranker) == (1200, 0, "")
    assert (default.chunk_size, default.chunk_overlap, default.reranker) == (1200, 0, "")
    assert table_for(base, "chunks_eval") == "chunks_eval"
    tables = {table_for(load_settings(env), "chunks_eval") for env in tool.CONFIGS.values()}
    assert len(tables) == 3  # 1200 (baseline), 600 and 400 chunk configs
    rr = [c for c, env in tool.CONFIGS.items() if "RERANKER" in env]
    assert sorted(rr) == ["c1200_rr", "c400_rr", "c600_rr"]
    assert set(tool.JUDGES) == {"small", "large"}


def test_generate_skips_when_meta_marks_the_run_complete(tmp_path, monkeypatch, capsys):
    import sys

    tool = _tool()
    out = tmp_path / "answers_c1200_s0.jsonl"
    out.with_suffix(".meta.json").write_text("{}")
    argv = ["x", "generate", "--config", "c1200", "--labeled", "l.jsonl", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv + ["--seed", "0", "--temperature", "0.3"])
    assert tool.main() == 0
    assert "skip" in capsys.readouterr().out
