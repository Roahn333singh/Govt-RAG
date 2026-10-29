"""One JSON line per request. No question text, headers, or credentials."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid

from opentelemetry import trace

from app.request_context import begin_request_fields, end_request_fields

_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCP_PROJECT")
    or "project-0ff01dde-a087-4c96-b04"
)
_lock = threading.Lock()
_in_flight = 0


def instance_id() -> str:
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.read().decode().strip()[:12] or "local"
    except Exception:
        return "local"


def install_request_logging(app) -> None:
    current_instance = instance_id()
    revision = os.getenv("K_REVISION", "local")

    @app.middleware("http")
    async def log_request(request, call_next):
        global _in_flight
        with _lock:
            _in_flight += 1
            entered = _in_flight
        request_id = uuid.uuid4().hex
        token = begin_request_fields()
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            fields = end_request_fields(token)
            with _lock:
                _in_flight -= 1
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            timings = fields.get("timings") or {}
            line = {
                "severity": "ERROR" if status >= 500 else "INFO",
                "message": "request_done",
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status": status,
                "duration_ms": duration_ms,
                "instance_id": current_instance,
                "revision": revision,
                "pid": os.getpid(),
                "thread": threading.current_thread().name,
                "in_flight": entered,
                "cache_hit": bool(fields.get("cache_hit")),
                "question_length": fields.get("question_length"),
                "rewrite_ms": timings.get("rewrite_ms", 0),
                "embed_ms": timings.get("embed_ms", 0),
                "qdrant_ms": timings.get("qdrant_ms", 0),
                "rerank_ms": timings.get("rerank_ms", 0),
                "generate_ms": timings.get("generate_ms", 0),
            }
            span_ctx = trace.get_current_span().get_span_context()
            if span_ctx is not None and span_ctx.trace_id:
                trace_id = format(span_ctx.trace_id, "032x")
                if trace_id != "0" * 32:
                    line["logging.googleapis.com/trace"] = (
                        f"projects/{_PROJECT}/traces/{trace_id}"
                    )
                    line["logging.googleapis.com/spanId"] = format(span_ctx.span_id, "016x")
            print(json.dumps(line, separators=(",", ":")), flush=True)
