"""LangSmith setup stays off without a key; run config carries cost metadata."""

import os

import pytest


def test_setup_stays_off_without_api_key(monkeypatch):
    from app import langsmith_tracing as ls

    ls.reset_langsmith_for_tests()
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert ls.setup_langsmith() is False
    assert ls.langsmith_enabled() is False
    assert os.getenv("LANGSMITH_TRACING") == "false"


def test_setup_auto_enables_when_key_present(monkeypatch):
    from app import langsmith_tracing as ls

    ls.reset_langsmith_for_tests()
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_not_a_real_key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert ls.setup_langsmith() is True
    assert ls.langsmith_enabled() is True
    assert os.getenv("LANGSMITH_PROJECT") == "govt-rag"
    assert os.getenv("LANGSMITH_TRACING") == "true"


def test_explicit_false_wins_even_with_key(monkeypatch):
    from app import langsmith_tracing as ls

    ls.reset_langsmith_for_tests()
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_not_a_real_key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert ls.setup_langsmith() is False
    assert ls.langsmith_enabled() is False


def test_graph_run_config_has_openrouter_cost_hooks(monkeypatch):
    from app.langsmith_tracing import graph_run_config

    monkeypatch.setenv("RERANKER", "gte")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv("GIT_SHA", "abc1234")
    cfg = graph_run_config(
        session_id="sess-1",
        language="hi",
        request_id="req-9",
        cached=False,
        stream=True,
    )
    assert cfg["configurable"]["thread_id"] == "sess-1"
    assert "lang:hi" in cfg["tags"]
    assert "stream" in cfg["tags"]
    assert "cache_miss" in cfg["tags"]
    assert cfg["metadata"]["ls_provider"] == "openrouter"
    assert cfg["metadata"]["ls_model_name"] == "google/gemini-2.5-flash"
    assert cfg["metadata"]["reranker"] == "gte"
    assert cfg["metadata"]["request_id"] == "req-9"
    assert cfg["metadata"]["git_sha"] == "abc1234"


def test_record_llm_usage_reads_openrouter_cost():
    from types import SimpleNamespace

    from app.langsmith_tracing import record_llm_usage

    response = SimpleNamespace(
        usage_metadata={"input_tokens": 10, "output_tokens": 4},
        response_metadata={"usage": {"cost": 0.00012}},
    )
    payload = record_llm_usage(response, "google/gemini-2.5-flash", role="generator")
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 4
    assert payload["total_cost"] == pytest.approx(0.00012)
