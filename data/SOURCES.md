# Corpus sources

| Field | Value |
|---|---|
| Origin | https://github.com/fastapi/fastapi, `docs/en/docs/tutorial/*.md` (top level, 40 pages) |
| Pinned commit | `69ddc2badf9da1123bba4d360225ae6b085985f6` |
| License | MIT, Copyright (c) 2018 Sebastián Ramírez (copied to `data/corpus/LICENSE` on fetch) |
| Retrieved | 2026-10-22 |
| Modifications | none to the files; at ingest the chunker drops `{* include *}` lines, `///` admonition markers and heading anchors |

The pages are not committed (several exceed this repo's 350-line file cap and they are
upstream content); `python tools/fetch_corpus.py` downloads exactly this list at the pinned commit.
