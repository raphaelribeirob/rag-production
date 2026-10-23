"""Download the FastAPI tutorial docs (MIT) at a pinned upstream commit into data/corpus/.

Pages are used unchanged. See data/SOURCES.md for origin, license and retrieval date.
"""

import sys
import urllib.request
from pathlib import Path

COMMIT = "69ddc2badf9da1123bba4d360225ae6b085985f6"
BASE = f"https://raw.githubusercontent.com/fastapi/fastapi/{COMMIT}"
PAGES = [
    "background-tasks.md",
    "bigger-applications.md",
    "body-fields.md",
    "body-multiple-params.md",
    "body-nested-models.md",
    "body-updates.md",
    "body.md",
    "cookie-param-models.md",
    "cookie-params.md",
    "cors.md",
    "debugging.md",
    "encoder.md",
    "extra-data-types.md",
    "extra-models.md",
    "first-steps.md",
    "frontend.md",
    "handling-errors.md",
    "header-param-models.md",
    "header-params.md",
    "index.md",
    "metadata.md",
    "middleware.md",
    "path-operation-configuration.md",
    "path-params-numeric-validations.md",
    "path-params.md",
    "query-param-models.md",
    "query-params-str-validations.md",
    "query-params.md",
    "request-files.md",
    "request-form-models.md",
    "request-forms-and-files.md",
    "request-forms.md",
    "response-model.md",
    "response-status-code.md",
    "schema-extra-example.md",
    "server-sent-events.md",
    "sql-databases.md",
    "static-files.md",
    "stream-json-lines.md",
    "testing.md",
]


def fetch(dest: Path = Path("data/corpus")) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    targets = [(f"docs/en/docs/tutorial/{p}", p) for p in PAGES] + [("LICENSE", "LICENSE")]
    for src, name in targets:
        out = dest / name
        if out.exists():
            continue
        with urllib.request.urlopen(f"{BASE}/{src}", timeout=30) as r:  # noqa: S310 (fixed host)
            out.write_bytes(r.read())
    print(f"corpus ok: {len(PAGES)} pages at fastapi@{COMMIT[:7]} in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(fetch())
