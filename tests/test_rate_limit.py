from types import SimpleNamespace

from agent.security import real_client_ip
from tests.conftest import API_HEADERS


def _req(xff=None, host="10.0.0.1"):
    headers = {"x-forwarded-for": xff} if xff else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))


def test_rightmost_xff_hop_is_the_key():
    assert real_client_ip(_req("1.1.1.1, 9.9.9.9")) == "9.9.9.9"
    assert real_client_ip(_req("2.2.2.2, 9.9.9.9")) == "9.9.9.9"


def test_no_xff_falls_back_to_socket_peer():
    assert real_client_ip(_req(None, host="10.1.2.3")) == "10.1.2.3"


def test_spoofed_left_hops_share_one_bucket(client):
    ok = limited = 0
    for i in range(20):
        r = client.post(
            "/chat",
            json={"message": f"What is the IGRS portal question {i}?"},
            headers={**API_HEADERS, "X-Forwarded-For": f"203.0.113.{i}, 198.51.100.7"},
        )
        if r.status_code == 200:
            ok += 1
        elif r.status_code == 429:
            limited += 1
    assert ok == 15
    assert limited == 5
