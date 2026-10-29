"""Per-request fields the nodes fill in and the access log reads.

The dict is created by the logging middleware. Mutating it is safe from a
worker thread because the thread holds the same dict, not a copy of the values.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_fields: ContextVar[dict | None] = ContextVar("request_fields", default=None)


def begin_request_fields() -> Token:
    return _fields.set(
        {"cache_hit": False, "question_length": None, "timings": {}}
    )


def end_request_fields(token: Token) -> dict:
    fields = _fields.get() or {"cache_hit": False, "question_length": None, "timings": {}}
    _fields.reset(token)
    return fields


def note_question_length(length: int) -> None:
    fields = _fields.get()
    if fields is not None:
        fields["question_length"] = length


def note_cache(hit: bool) -> None:
    fields = _fields.get()
    if fields is not None:
        fields["cache_hit"] = hit


def note_timing(name: str, elapsed_ms: float) -> None:
    fields = _fields.get()
    if fields is None:
        return
    timings = fields["timings"]
    # Parallel searches report the same step at once. Keep the slowest one,
    # which is the wall-clock contribution, rather than adding them together.
    timings[name] = round(max(float(timings.get(name, 0.0)), elapsed_ms), 2)
