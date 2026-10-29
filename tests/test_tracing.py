from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.telemetry import run_traced_in_threadpool, setup_telemetry


def test_telemetry_stays_off_unless_enabled():
    from fastapi import FastAPI

    assert setup_telemetry(FastAPI()) is False


async def test_threadpool_span_stays_under_its_parent():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    try:
        tracer = trace.get_tracer("test-nesting")

        def child():
            with tracer.start_as_current_span("rerank"):
                return "ok"

        with tracer.start_as_current_span("node.retriever"):
            assert await run_traced_in_threadpool(child) == "ok"
    finally:
        trace.set_tracer_provider(previous)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["rerank"].parent.span_id == spans["node.retriever"].context.span_id
