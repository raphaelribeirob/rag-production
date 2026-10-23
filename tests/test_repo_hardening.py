"""Repo-level security controls (see SECURITY.md OWASP mapping)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
SECRET_KEYS = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD)", re.I)
PLACEHOLDERS = ("", "change-me-to-a-random-value")


def test_actions_pinned_by_full_sha():
    assert WORKFLOWS
    for wf in WORKFLOWS:
        for ref in re.findall(r"uses:\s*(\S+)", wf.read_text()):
            assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), f"{wf.name}: {ref}"


def test_workflow_token_is_read_only():
    for wf in WORKFLOWS:
        text = wf.read_text()
        assert re.search(r"^permissions:\s*\n\s+contents:\s*read\s*$", text, re.M), wf.name
        assert "write" not in text.split("jobs:")[0], wf.name
        assert "persist-credentials: false" in text, wf.name


def test_env_example_has_only_placeholder_secrets():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if SECRET_KEYS.search(key):
            assert value.strip() in PLACEHOLDERS, f"{key} must be a placeholder"


def test_dotenv_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()


def test_corpus_source_pinned_to_full_commit_and_https():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fc", ROOT / "tools" / "fetch_corpus.py")
    fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc)
    assert re.fullmatch(r"[0-9a-f]{40}", fc.COMMIT)
    assert fc.BASE.startswith("https://raw.githubusercontent.com/fastapi/fastapi/")
    assert all(re.fullmatch(r"[a-z0-9-]+\.md", p) for p in fc.PAGES)
