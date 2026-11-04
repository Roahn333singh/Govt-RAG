"""QA API key skips chat rate limits; normal callers and bad keys do not."""

from __future__ import annotations

import json

from tests.conftest import API_HEADERS

Q = {"message": "What is the URL for the Pragati portal?"}


def test_qa_key_is_not_rate_limited(client, monkeypatch, fakes):
    monkeypatch.setenv("QA_API_KEY", "qa-loadtest-key")
    fakes.main.limiter.reset()
    statuses = []
    for i in range(20):
        r = client.post(
            "/chat",
            json={"message": f"What is the Pragati portal question {i}?"},
            headers={"X-API-Key": "qa-loadtest-key"},
        )
        statuses.append(r.status_code)
    assert statuses.count(200) == 20
    assert 429 not in statuses


def test_normal_key_still_429_after_15(client, fakes):
    fakes.main.limiter.reset()
    ok = limited = 0
    for i in range(20):
        r = client.post(
            "/chat",
            json={"message": f"What is the IGRS portal question {i}?"},
            headers={**API_HEADERS, "X-Forwarded-For": "198.51.100.42"},
        )
        if r.status_code == 200:
            ok += 1
        elif r.status_code == 429:
            limited += 1
    assert ok == 15
    assert limited == 5


def test_wrong_key_is_401(client, fakes):
    fakes.main.limiter.reset()
    r = client.post("/chat", json=Q, headers={"X-API-Key": "definitely-not-a-key"})
    assert r.status_code == 401


def test_qa_run_id_does_not_leak_key(client, monkeypatch, fakes, capsys):
    monkeypatch.setenv("QA_API_KEY", "qa-loadtest-key")
    fakes.main.limiter.reset()
    r = client.post(
        "/chat",
        json=Q,
        headers={"X-API-Key": "qa-loadtest-key", "X-QA-Run-Id": "load-abc"},
    )
    assert r.status_code == 200
    lines = [line for line in capsys.readouterr().out.splitlines() if '"request_done"' in line]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["key_id"] == "qa"
    assert "qa-loadtest-key" not in lines[-1]


def test_session_token_is_not_qa_exempt(client, fakes, monkeypatch):
    """Even with QA_API_KEY set, a session token still consumes the IP budget."""
    from agent.security import issue_session_token

    monkeypatch.setenv("QA_API_KEY", "qa-loadtest-key")
    fakes.main.limiter.reset()
    token = issue_session_token("testclient")
    ok = limited = 0
    for i in range(20):
        r = client.post(
            "/chat",
            json={"message": f"Session limited question {i}?"},
            headers={"X-Session-Token": token},
        )
        if r.status_code == 200:
            ok += 1
        elif r.status_code == 429:
            limited += 1
    assert ok == 15
    assert limited == 5
