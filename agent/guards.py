"""Input / output guards for the public Manual AI RAG desk.

- Cap message length
- Detect prompt-injection patterns (never call LLM / never cache)
- Filter leaked system-prompt or raw context dumps from answers
- Lexical overlap helper for semantic-cache precision
"""

from __future__ import annotations

import re
from typing import Iterable

MAX_MESSAGE_CHARS = 2000
PROMPT_VERSION = "v3-hardened-2026-10"

REFUSAL_TEXT = (
    "I am an AI assistant specialized in UP Government Data. "
    "I cannot answer that request."
)

INJECTION_REFUSAL = (
    "I can only answer questions about UP Government manuals and orders. "
    "I will not follow instructions that ask me to ignore rules, "
    "reveal my prompt, or invent answers."
)

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
        r"forget\s+(all\s+)?(your\s+)?(rules|instructions|guidelines)",
        r"you\s+are\s+now\s+(?:DAN|jailbroken|unrestricted)",
        r"repeat\s+(verbatim\s+)?(the\s+)?(full\s+)?(text\s+of\s+)?(the\s+)?(instructions?|rules|system\s+prompt)",
        r"print\s+(the\s+)?(raw\s+)?(context|system\s+prompt|instructions?)",
        r"output\s+exactly\s*:",
        r"optimized\s+search\s+query\s*:",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
        r"dump\s+(the\s+)?(context|prompt|instructions?)",
        r"<\s*/?\s*system\s*>",
        r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
        r"qa-probe[-_]?\d*",
        r"finish\s+with\s+the\s+exact\s+line",
        r"end\s+(your\s+)?answer\s+with\s+(the\s+)?exact\s+(line|phrase|words?)",
        r"end\s+(your\s+)?answer\s+with\s+the\s+exact\s+line",
        r"answer\s+with\s+just\s+the\s+single\s+word",
        r"reply\s+only\s+with\s+the\s+word",
        r"quote\s+the\s+full\s+raw\s+text",
        r"word\s+for\s+word\s+the\s+rules\s+you\s+were\s+given",
        r"\bMANGOK\w*\b",
        r"\b[A-Z]{4,}\d{2,}[A-Z0-9]*\s+CACHED\b",
    )
]

_OUTPUT_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"CRITICAL RULES FOR FACTUAL ACCURACY",
        r"Formatting Rules for UI Compatibility",
        r"You are a (highly precise|search-query optimizer)",
        r"^Context:\s*$",
        r"Document:\s*.+\nPage:\s*\d+\nText:",
        r"Optimized Search Query:",
    )
]

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def detect_answer_language(text: str) -> str:
    """Devanagari in the question → Hindi; otherwise English."""
    if not text:
        return "en"
    return "hi" if _DEVANAGARI_RE.search(text) else "en"


def is_injection(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def validate_message_length(text: str) -> None:
    """Raise ValueError if over the hard cap (caller maps to HTTP 413)."""
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(
            f"Message too long ({len(text)} chars). "
            f"Maximum is {MAX_MESSAGE_CHARS} characters."
        )


# Rare ALLCAPS+digit tokens callers use as cache-poison markers
_MARKER_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6,}\b")
_INSTRUCTION_ECHO_RE = re.compile(
    r"(?i)\b(CACHED|BANANA|MANGOK\w*|QA-PROBE[-_]?\w*)\b"
)


def extract_caller_markers(text: str) -> set[str]:
    """Tokens in the user message that look like audit/poison markers."""
    if not text:
        return set()
    found = {m.group(0).upper() for m in _MARKER_TOKEN_RE.finditer(text)}
    found |= {m.group(0).upper() for m in _INSTRUCTION_ECHO_RE.finditer(text)}
    return found


def answer_echoes_markers(question: str, answer: str) -> bool:
    """True if the answer repeats a caller-supplied marker (soft cache poison)."""
    if not answer:
        return False
    markers = extract_caller_markers(question)
    if not markers:
        # Still block known poison echoes even if the question lacked them
        return bool(_INSTRUCTION_ECHO_RE.search(answer))
    upper = answer.upper()
    return any(m in upper for m in markers)


def filter_output(answer: str, question: str | None = None) -> str | None:
    """Return None if the answer looks like a prompt/context leak or marker echo."""
    if not answer:
        return None
    for p in _OUTPUT_LEAK_PATTERNS:
        if p.search(answer):
            return None
    # Heuristic: many "Document:" / "Page:" lines → raw context dump
    if answer.count("Document:") >= 3 and answer.count("Page:") >= 3:
        return None
    if question and answer_echoes_markers(question, answer):
        return None
    if _INSTRUCTION_ECHO_RE.search(answer):
        return None
    return answer


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def lexical_overlap(a: str, b: str) -> float:
    """Token Jaccard similarity in [0, 1]."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def has_shared_anchors(a: str, b: str) -> bool:
    """True if both share a number, URL-ish token, or portal name."""
    nums_a = set(re.findall(r"\d+(?:\.\d+)?", a or ""))
    nums_b = set(re.findall(r"\d+(?:\.\d+)?", b or ""))
    if nums_a & nums_b:
        return True
    portals = (
        "pragati",
        "igrs",
        "fmisc",
        "upsdc",
        "निविदा",
        "अनुच्छेद",
        "article",
    )
    al, bl = (a or "").lower(), (b or "").lower()
    return any(p in al and p in bl for p in portals)


def cache_lexical_ok(query: str, cached_question: str) -> bool:
    return lexical_overlap(query, cached_question) >= 0.5 or has_shared_anchors(
        query, cached_question
    )


def sanitize_rewrite(raw: str, fallback: str, max_words: int = 30) -> str:
    """Keep first line, strip spoofed prefixes, cap length."""
    if not raw:
        return fallback
    line = raw.strip().splitlines()[0].strip()
    line = re.sub(
        r"^(optimized\s+search\s+query\s*:?\s*)",
        "",
        line,
        flags=re.IGNORECASE,
    ).strip()
    # Reject if the rewrite itself looks like an injection payload
    if is_injection(line):
        return fallback
    words = line.split()
    if not words:
        return fallback
    if len(words) > max_words:
        line = " ".join(words[:max_words])
    return line


def sources_from_hits(hits: Iterable, max_chars: int = 1500) -> list[dict]:
    """Map Qdrant hit payloads → UI-friendly source dicts."""
    out: list[dict] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        text = (payload.get("text") or "")[:max_chars]
        out.append(
            {
                "source_document": payload.get("document-id")
                or payload.get("source")
                or "Unknown",
                "page_start": payload.get("page"),
                "chapter": payload.get("chapter"),
                "section": payload.get("section"),
                "content": text,
                "score": float(getattr(hit, "score", 0.0) or 0.0),
            }
        )
    return out
