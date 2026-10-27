"""When a semantic-cache entry may be served, and which answers may be stored."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime

from agent.guards import PROMPT_VERSION, lexical_overlap

CACHE_SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.95"))
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "168"))
CACHE_MIN_JACCARD = float(os.getenv("CACHE_MIN_JACCARD", "0.5"))

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_ENTITIES = (
    "pragati",
    "igrs",
    "jan-sunwai",
    "jansunwai",
    "fmisc",
    "upsdc",
    "hello kisan",
    "gpf",
    "rti",
    "adcp",
    "dpr",
    "nivida",
    "निविदा",
    "article",
    "अनुच्छेद",
    "idt-1",
    "idt-2",
    "idt-3",
)
_PREMISE_CORRECTION_RE = re.compile(
    r"\bnot\s+(?:₹\s*|rs\.?\s*)?\d|\bactually\b|\bincorrect\s+premise\b|\bfalse\s+premise\b|"
    r"\bcontrary\s+to\b|\bis\s+incorrect\b|\bnot\s+correct\b|\bmistaken\b|"
    r"सही\s+नहीं|गलत\s+है",
    re.IGNORECASE,
)


def doc_version() -> str:
    return os.getenv("DOC_VERSION", "1")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM_RE.findall(text or "")}


def entities(text: str) -> set[str]:
    low = (text or "").lower()
    return {e for e in _ENTITIES if e in low}


def questions_match(query: str, cached_question: str) -> bool:
    """Lexical gate on top of vector similarity: overlap AND same numbers/entities."""
    if lexical_overlap(query, cached_question) < CACHE_MIN_JACCARD:
        return False
    if numbers(query) != numbers(cached_question):
        return False
    return entities(query) == entities(cached_question)


def is_premise_correction(question: str, answer: str) -> bool:
    """Answers that push back on the question's premise are specific to that wording."""
    if _PREMISE_CORRECTION_RE.search(answer or ""):
        return True
    # The question asserts a figure the answer does not repeat -> likely a correction
    q_nums = numbers(question)
    return bool(q_nums) and not q_nums <= numbers(answer)


def entry_fresh(payload: dict, now: float | None = None) -> bool:
    if payload.get("prompt_version") != PROMPT_VERSION:
        return False
    if payload.get("doc_version") != doc_version():
        return False
    created = payload.get("created_at")
    if not created:
        return False
    try:
        if isinstance(created, (int, float)):
            ts = float(created)
        else:
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return False
    return ((now or time.time()) - ts) <= CACHE_TTL_HOURS * 3600
