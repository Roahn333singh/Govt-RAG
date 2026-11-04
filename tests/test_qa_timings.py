"""QA clients see stage timings; normal clients do not."""

from __future__ import annotations

import json

from tests.conftest import API_HEADERS

Q = {"message": "What is the URL for the Pragati portal?"}
EXPECTED_KEYS = {
    "embed_ms",
    "qdrant_ms",
    "rerank_ms",
    "llm_first_token_ms",
    "llm_total_ms",
}


def test_qa_key_gets_timings_in_body_and_header(client, monkeypatch, fakes):
    monkeypatch.setenv("QA_API_KEY", "qa-loadtest-key")
    fakes.main.limiter.reset()
    r = client.post("/chat", json=Q, headers={"X-API-Key": "qa-loadtest-key"})
    assert r.status_code == 200
    body = r.json()
    assert "timings" in body
    assert set(body["timings"]) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert isinstance(body["timings"][key], (int, float))
    header = r.headers.get("x-timing")
    assert header
    assert json.loads(header) == body["timings"]
    # Fake path still embeds the query before the graph, so embed_ms is real.
    assert body["timings"]["embed_ms"] > 0


def test_normal_key_has_no_timings(client, fakes):
    fakes.main.limiter.reset()
    r = client.post("/chat", json=Q, headers=API_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "timings" not in body
    assert r.headers.get("x-timing") is None


def test_request_done_logs_llm_timing_fields(client, monkeypatch, fakes, capsys):
    monkeypatch.setenv("QA_API_KEY", "qa-loadtest-key")
    fakes.main.limiter.reset()
    r = client.post("/chat", json=Q, headers={"X-API-Key": "qa-loadtest-key"})
    assert r.status_code == 200
    lines = [line for line in capsys.readouterr().out.splitlines() if '"request_done"' in line]
    assert lines
    payload = json.loads(lines[-1])
    assert "llm_first_token_ms" in payload
    assert "llm_total_ms" in payload
    assert "embed_ms" in payload
    assert "rerank_ms" in payload
    assert "qa-loadtest-key" not in lines[-1]
