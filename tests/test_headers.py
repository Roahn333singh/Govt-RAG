from tests.conftest import API_HEADERS

SECURITY_HEADERS = (
    "x-content-type-options",
    "referrer-policy",
    "x-frame-options",
    "content-security-policy",
    "strict-transport-security",
    "permissions-policy",
    "cross-origin-opener-policy",
)


def test_seven_security_headers(client):
    headers = {k.lower(): v for k, v in client.get("/health").headers.items()}
    for name in SECURITY_HEADERS:
        assert headers.get(name), name
    assert headers["x-content-type-options"] == "nosniff"
    assert "max-age=" in headers["strict-transport-security"]
    assert headers["cross-origin-opener-policy"] == "same-origin"


def test_overlong_message_is_413(client):
    r = client.post("/chat", json={"message": "a" * 2001}, headers=API_HEADERS)
    assert r.status_code == 413
    assert "Traceback" not in r.text


def test_html_tags_are_stripped(client, fakes):
    fakes.graph.answer = "<script>alert(1)</script><b>The limit is 500 KB.</b>\n\nSources: Doc.pdf, Page: 1"
    r = client.post("/chat", json={"message": "What is the IGRS limit?"}, headers=API_HEADERS)
    answer = r.json()["answer"]
    assert "<" not in answer
    assert "500 KB" in answer


def test_unhandled_error_hides_the_traceback(fakes):
    from fastapi.testclient import TestClient

    @fakes.main.app.get("/__test_boom")
    def boom():
        raise RuntimeError("secret internal detail")

    raw = TestClient(fakes.main.app, raise_server_exceptions=False)
    r = raw.get("/__test_boom")
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Internal error"
    assert body["request_id"]
    assert "secret" not in r.text
    assert "Traceback" not in r.text
