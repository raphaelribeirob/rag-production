# Data

**Corpus: FastAPI documentation** (English Markdown pages under `docs/en/docs/` of
https://github.com/fastapi/fastapi). We use a subset of about 50 to 200 pages.

**License and attribution:** the MIT License, Copyright (c) 2018 Sebastián Ramírez.
The upstream `LICENSE` file is copied to `data/corpus/LICENSE` along with the pages. The docs
are used unchanged, apart from normalization at ingest time.

Layout (planned):
- `data/corpus/`: source documents (.md / .pdf / .txt). Pin the upstream commit SHA in SOURCES.md.
- `data/SOURCES.md`: origin URL, license, retrieval date and commit hash for each source.

Rules: no private data, no credentials, no pages scraped under unclear terms.
