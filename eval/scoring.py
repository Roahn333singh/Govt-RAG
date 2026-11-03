"""Grader-only text matching. Not used by the serving path."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden200.csv"
SAMPLE25 = ROOT / "eval" / "review_sample25.csv"
PROD50 = ROOT / "eval" / "prod50.jsonl"

_PART_SEP = " || "


def split_parts(value: str | None) -> list[str]:
    """Split a grader field on the joiner used for multi-quote rows."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(_PART_SEP) if part.strip()]


def normalize_text(text: str | None) -> str:
    """NFKC, drop private-use glyphs and control characters, lowercase, collapse space.

    Newlines and tabs are whitespace, so a quote that wraps a line still matches.
    Other control characters and the private-use area (bullet glyphs such as
    U+F097) are removed.
    """
    raw = unicodedata.normalize("NFKC", text or "")
    kept: list[str] = []
    for ch in raw:
        if 0xE000 <= ord(ch) <= 0xF8FF:
            continue
        if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t":
            continue
        kept.append(ch)
    return re.sub(r"\s+", " ", "".join(kept).lower()).strip()


def sources_match(chunk_source: str | None, expected_docs: list[str]) -> bool:
    got = normalize_text(chunk_source)
    if not got:
        return False
    for doc in expected_docs:
        want = normalize_text(doc)
        if not want:
            continue
        if got == want or got.endswith("/" + want):
            return True
    return False


def part_found(part: str, chunks: list[dict], expected_docs: list[str]) -> bool:
    """True when a chunk from a matching source document contains the quote."""
    needle = normalize_text(part)
    if not needle or not expected_docs:
        return False
    for chunk in chunks:
        if not sources_match(chunk.get("source"), expected_docs):
            continue
        if needle in normalize_text(chunk.get("text")):
            return True
    return False


def reciprocal_rank(parts: list[str], chunks: list[dict], expected_docs: list[str]) -> float:
    if not parts or not expected_docs:
        return 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if not sources_match(chunk.get("source"), expected_docs):
            continue
        hay = normalize_text(chunk.get("text"))
        if any(normalize_text(part) and normalize_text(part) in hay for part in parts):
            return 1.0 / rank
    return 0.0


def stage_metrics(parts: list[str], chunks: list[dict], expected_docs: list[str]) -> dict:
    if not parts:
        return {"hit": None, "recall": None, "mrr": None, "source_hit": None, "k": len(chunks)}
    found = [part_found(part, chunks, expected_docs) for part in parts]
    return {
        "hit": any(found),
        "recall": sum(found) / len(found),
        "mrr": reciprocal_rank(parts, chunks, expected_docs),
        "source_hit": any(sources_match(chunk.get("source"), expected_docs) for chunk in chunks),
        "k": len(chunks),
    }


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_facts(value) -> list:
    text = (value or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _normalize_item(row: dict) -> dict:
    facts = row.get("must_include_facts")
    if isinstance(facts, list):
        must = [str(item) for item in facts]
    else:
        must = as_facts(facts)
    answerable = row.get("answerable")
    if not isinstance(answerable, bool):
        answerable = as_bool(answerable)
    return {
        "id": str(row["id"]).strip(),
        "question": row["question"],
        "language": (row.get("language") or "").strip(),
        "type": (row.get("type") or "").strip(),
        "difficulty": (row.get("difficulty") or "").strip(),
        "expected_answer": row.get("expected_answer") or "",
        "must_include_facts": must,
        "source_docs": split_parts(row.get("source_doc")),
        "evidence_parts": split_parts(row.get("evidence_quote")),
        "answerable": answerable,
    }


def load_items(which: str = "full") -> list[dict]:
    """Load the golden set. `sample25` / `prod50` are the named review files."""
    import csv

    if which == "prod50":
        items = []
        for line in PROD50.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(_normalize_item(json.loads(line)))
        return items

    path = SAMPLE25 if which == "sample25" else GOLDEN
    items: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            items.append(_normalize_item(row))
    return items
