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

# Only instruction-override / output-control phrasing. Never domain words
# ("rules", "penalty", "disciplinary", "instructions for filling a form").
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above|earlier|preceding|system|original)\s+(instructions?|guidance|rules|prompts?|directions)",
        r"\b(forget|ignore|disregard)\s+(all\s+)?(your|the)\s+(rules|instructions|guidelines|guardrails)\b",
        r"\byou\s+are\s+now\s+(?:DAN|jailbroken|unrestricted|in\s+developer\s+mode)",
        r"\bdeveloper\s+mode\b",
        r"\b(repeat|print|show|reveal|output|dump|display|quote|leak)\b.{0,40}\b(system\s+prompt|your\s+(instructions|rules|prompt)|the\s+(instructions|rules)\s+you\s+(were|have\s+been)\s+given|hidden\s+(instructions|prompt))",
        r"\b(print|dump|quote|output|show)\b.{0,30}\braw\s+(context|text|chunks?)",
        r"\bword\s+for\s+word\b.{0,40}\b(rules|instructions|prompt|context)",
        r"\b(instructions?|rules|guidelines)\b[^.?!]{0,30}\byou\s+(were|have\s+been)\s+given\b",
        r"\bsystem\s+prompt\b",
        r"\boutput\s+exactly\s*:",
        r"\boptimized\s+search\s+query\s*:",
        r"\breply\s+(only\s+)?with\s+(only\s+)?(the\s+)?(single\s+)?word\b",
        r"\banswer\s+with\s+(just|only)\s+the\s+(single\s+)?word\b",
        r"\b(end|finish|close|terminate)\s+(your\s+)?(answer|reply|response)?\s*with\s+(the\s+)?exact\s+(line|phrase|words?|text)",
        r"\bappend\s+(the\s+)?(exact\s+)?(line|phrase|marker|token)\b",
        r"<\s*/?\s*(system|assistant|im_start|im_end)\s*\|?>",
        r"<\|\s*(system|assistant|im_start|im_end)\s*\|>",
        r"^\s*(system|assistant)\s*:",
        r"\bact\s+as\s+(if\s+you\s+have\s+no|an?\s+unrestricted)",
        r"\bpretend\s+(that\s+)?you\s+(have\s+no|are\s+not\s+bound)",
        r"\bqa-probe[-_]?\w*",
    )
]

_OUTPUT_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
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


# Known poison tokens from past probes; never legitimate in an answer.
_POISON_TOKEN_RE = re.compile(r"\b(BANANA|MANGOK\w*|QA-PROBE[-_]?\w*)\b", re.IGNORECASE)
# Questions asking the model to emit a specific string in its answer
_OUTPUT_CONTROL_RE = re.compile(
    r"\b(end|finish|close)\s+(your\s+)?(answer|reply|response)?\s*with|\bappend\b|"
    r"\binclude\s+the\s+(exact\s+)?(line|word|phrase)|\breply\s+(only\s+)?with|\bsay\s+only\b",
    re.IGNORECASE,
)
_MARKER_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{5,}\b")


def extract_caller_markers(text: str) -> set[str]:
    """ALLCAPS+digit tokens the caller asked the model to emit."""
    if not text or not _OUTPUT_CONTROL_RE.search(text):
        return set()
    return {m.group(0).upper() for m in _MARKER_TOKEN_RE.finditer(text)}


def answer_echoes_markers(question: str, answer: str) -> bool:
    """True if the answer repeats a caller-requested marker or a poison token."""
    if not answer:
        return False
    if _POISON_TOKEN_RE.search(answer):
        return True
    upper = answer.upper()
    return any(m in upper for m in extract_caller_markers(question or ""))


def filter_output(answer: str, question: str | None = None) -> str | None:
    """Return None if the answer looks like a prompt/context leak or marker echo.

    A "Sources:" block listing many Document/Page lines is normal; only the raw
    context layout (Document / Page / Text) counts as a dump.
    """
    if not answer:
        return None
    for p in _OUTPUT_LEAK_PATTERNS:
        if p.search(answer):
            return None
    if answer.count("Text:") >= 2 and answer.count("Document:") >= 2:
        return None
    if answer_echoes_markers(question or "", answer):
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
