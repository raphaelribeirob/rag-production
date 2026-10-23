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
