import time

from agent.security import issue_session_token, verify_session_token
from tests.conftest import API_HEADERS

Q = {"message": "What is the URL for the Pragati portal?"}


def test_chat_without_key_is_401(client):
    r = client.post("/chat", json=Q)
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


def test_chat_with_wrong_key_is_401(client):
    assert client.post("/chat", json=Q, headers={"X-API-Key": "nope"}).status_code == 401


def test_chat_with_key_is_200(client):
    r = client.post("/chat", json=Q, headers=API_HEADERS)
    assert r.status_code == 200
    assert r.json()["answer"]


def test_stream_requires_auth(client):
    assert client.post("/chat/stream", json=Q).status_code == 401


def test_ui_session_token_flow(client):
    r = client.get("/ui/session", headers={"Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200
    token = r.json()["token"]
    r2 = client.post("/chat", json=Q, headers={"X-Session-Token": token})
    assert r2.status_code == 200


def test_ui_session_refused_cross_site(client):
    assert client.get("/ui/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/ui/session").status_code == 403


def test_token_bound_to_ip_and_expiry():
    tok = issue_session_token("1.2.3.4")
    assert verify_session_token(tok, "1.2.3.4")
    assert not verify_session_token(tok, "5.6.7.8")
    assert not verify_session_token(tok, "1.2.3.4", now=time.time() + 10_000)
    assert not verify_session_token("garbage", "1.2.3.4")


def test_cors_preflight_from_evil_origin_rejected(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_cors_preflight_from_allowed_origin(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "https://manual-ai-rag-nszha5soya-ey.a.run.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert r.status_code == 200
