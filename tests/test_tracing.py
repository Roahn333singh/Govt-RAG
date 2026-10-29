import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.telemetry import run_traced_in_threadpool, setup_telemetry
from tests.conftest import API_HEADERS


def test_request_log_records_length_not_text(client, capsys):
    question = "SECRET-QUESTION-TEXT about the Pragati portal"
    response = client.post("/chat", json={"message": question}, headers=API_HEADERS)
    assert response.headers["x-request-id"]
    lines = [line for line in capsys.readouterr().out.splitlines() if '"request_done"' in line]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["question_length"] == len(question)
    assert payload["path"] == "/chat"
    assert "duration_ms" in payload
    assert payload["instance_id"]
    assert "SECRET-QUESTION-TEXT" not in lines[-1]
    assert "authorization" not in lines[-1].lower()
    assert "api-key" not in lines[-1].lower()


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
