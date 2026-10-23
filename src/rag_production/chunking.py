"""Heading-aware Markdown chunker for the FastAPI docs corpus."""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(\{\s*#[^}]*\})?\s*$")
_INCLUDE = re.compile(r"^\{\*.*\*\}\s*$")
_ADMONITION = re.compile(r"^///")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    heading: str
    text: str


def _sections(text: str) -> list[tuple[list[str], list[str]]]:
    """Return (heading path, body lines) per section; ignores '#' lines inside code fences."""
    path: list[str] = []
    out: list[tuple[list[str], list[str]]] = []
    body: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            if body and any(b.strip() for b in body) or (path and not out):
                out.append((list(path), body))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            body = []
            continue
        if _INCLUDE.match(line) or _ADMONITION.match(line):
            continue
        body.append(line)
    if body and any(b.strip() for b in body) or (path and not out):
        out.append((list(path), body))
    return [(p, b) for p, b in out if any(x.strip() for x in b)]


def _split(body: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:  # hard-wrap pathological paragraphs
            parts.append(para[:max_chars])
            para = para[max_chars:]
        if cur and len(cur) + len(para) + 2 > max_chars:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def _tail(text: str, n: int) -> str:
    """Last <= n chars of text, starting at a word boundary when possible."""
    if n <= 0 or not text:
        return ""
    tail = text[-n:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 and len(text) > n else tail


def chunk_markdown(doc_id: str, text: str, max_chars: int = 1200, overlap: int = 0) -> list[Chunk]:
    """Split by heading, then pack paragraphs up to max_chars (heading prefix included).

    overlap > 0 starts each piece after the first in a section with the last <= overlap chars of
    the previous piece (word-aligned); pieces are packed to max_chars - overlap so the total
    still fits. overlap=0 reproduces the original chunker exactly.
    """
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    chunks: list[Chunk] = []
    for path, body in _sections(text):
        heading = " > ".join(path) if path else doc_id
        budget = max(200, max_chars - len(heading) - 2)
        if overlap and overlap >= budget - 1:
            raise ValueError(f"overlap {overlap} must be smaller than the chunk budget {budget}")
        pieces = _split("\n".join(body), budget - (overlap + 1 if overlap else 0))
        for i, piece in enumerate(pieces):
            if overlap and i:
                piece = f"{_tail(pieces[i - 1], overlap)} {piece}"
            cid = f"{doc_id}#{len(chunks)}"
            chunks.append(Chunk(cid, doc_id, heading, f"{heading}\n\n{piece}"))
    return chunks
