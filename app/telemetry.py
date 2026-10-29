"""OpenTelemetry setup. Does nothing unless OTEL_ENABLED=true."""

from __future__ import annotations

import os

from opentelemetry import context as otel_context
from opentelemetry import trace
from starlette.concurrency import run_in_threadpool


def otel_enabled() -> bool:
    return os.getenv("OTEL_ENABLED") == "true"


def setup_telemetry(app) -> bool:
    """Configure Cloud Trace export and auto-instrument FastAPI and httpx."""
    if not otel_enabled():
        return False

    from opentelemetry import propagate
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.propagators.cloud_trace_propagator import (
        CloudTraceFormatPropagator,
    )
    from opentelemetry.resourcedetector.gcp_resource_detector import (
        GoogleCloudResourceDetector,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    rate = float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))
    resource = Resource.create(
        {
            "service.name": "manual-ai-rag",
            "service.version": os.getenv("GIT_SHA", "dev"),
        }
    )
    try:
        resource = resource.merge(GoogleCloudResourceDetector().detect())
    except Exception as exc:
        print(f"otel: GCP resource detector skipped: {type(exc).__name__}")

    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(rate)),
    )
    provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
    trace.set_tracer_provider(provider)
    # Join the trace Cloud Run already put in X-Cloud-Trace-Context.
    propagate.set_global_textmap(CloudTraceFormatPropagator())

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ui")
    HTTPXClientInstrumentor().instrument()
    print(f"otel: Cloud Trace export on (sample rate {rate})")
    return True


def run_traced_in_threadpool(fn, *args, **kwargs):
    """Run fn on the thread pool without dropping the current trace context.

    asyncio.to_thread copies context on 3.11, but Starlette's threadpool does
    not. Spans opened inside fn would otherwise be orphaned from the request.
    """
    ctx = otel_context.get_current()

    def _call():
        token = otel_context.attach(ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return run_in_threadpool(_call)
