from rag_production.chunking import chunk_markdown

DOC = """# Path Parameters { #path-parameters }

Intro about path params.

{* ../../docs_src/path_params/tutorial001_py310.py hl[6:7] *}

## With types { #with-types }

Use type annotations for `item_id`.
"""


def test_splits_on_headings_and_strips_anchors_and_includes():
    chunks = chunk_markdown("path-params.md", DOC)
    assert [c.heading for c in chunks] == ["Path Parameters", "Path Parameters > With types"]
    assert all("{ #" not in c.text and "{*" not in c.text for c in chunks)
    assert chunks[1].text.startswith("Path Parameters > With types")


def test_ids_are_stable_and_unique():
    a = chunk_markdown("p.md", DOC)
    b = chunk_markdown("p.md", DOC)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] == ["p.md#0", "p.md#1"]
    assert all(c.doc_id == "p.md" for c in a)


def test_long_section_split_under_budget():
    text = "# T\n\n" + "\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(30))
    chunks = chunk_markdown("t.md", text, max_chars=800)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 + 100 for c in chunks)


def test_fenced_code_heading_like_lines_are_not_headings():
    text = "# T\n\n```bash\n# not a heading\n$ pip install fastapi\n```\n"
    chunks = chunk_markdown("t.md", text)
    assert len(chunks) == 1
    assert "pip install fastapi" in chunks[0].text


LONG = "# T\n\n" + "\n\n".join(f"para{i} " + "word " * 30 for i in range(12))


def test_default_chunking_is_unchanged_by_the_new_parameters():
    assert chunk_markdown("t.md", LONG) == chunk_markdown("t.md", LONG, max_chars=1200, overlap=0)


def test_smaller_chunks_with_overlap_stay_under_size_and_share_text():
    small = chunk_markdown("t.md", LONG, max_chars=400, overlap=80)
    assert len(small) >= len(chunk_markdown("t.md", LONG, max_chars=400)) > 1
    assert all(len(c.text) <= 400 for c in small)
    for prev, nxt in zip(small, small[1:], strict=False):
        body = nxt.text.split("\n\n", 1)[1]
        assert body[:20] in prev.text  # each piece starts with the tail of the previous one


def test_overlap_must_be_smaller_than_the_chunk_budget():
    import pytest

    with pytest.raises(ValueError):
        chunk_markdown("t.md", LONG, max_chars=300, overlap=300)
