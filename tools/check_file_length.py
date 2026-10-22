"""Fail if any tracked text/source file exceeds MAX_LINES (default 350).

Policy from vibe-coding-toolkit prompts 07/08 (Matheus Gomes, MIT, commit 790a895).
"""

import os
import subprocess
import sys

EXTS = (".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".jsonl", ".json")
SKIP: tuple[str, ...] = ()


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f.endswith(EXTS)]


def main() -> int:
    limit = int(os.environ.get("MAX_LINES", "350"))
    over = []
    for path in tracked_files():
        with open(path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        if n > limit and path not in SKIP:
            over.append((n, path))
    for n, path in sorted(over, reverse=True):
        print(f"{path}: {n} lines (max {limit})", file=sys.stderr)
    if over:
        return 1
    print(f"file-length ok (max {limit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
