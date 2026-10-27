from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from qdrant_client.models import PointStruct

from agent.cache_policy import (
    entry_fresh,
    is_premise_correction,
    questions_match,
)
from agent.guards import PROMPT_VERSION
from tests.conftest import API_HEADERS

ADMIN = {"X-Admin-Token": "test-admin-token"}


def _payload(question, answer="Cached answer.", **over):
    p = {
        "question": question,
        "answer": answer,
        "language": "en",
        "sources": [{"source_document": "Doc.pdf", "page_start": 1}],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "doc_version": "test-docs-1",
    }
    p.update(over)
    return p


def test_number_mismatch_is_a_miss():
    assert not questions_match(
        "Is the IGRS upload limit 50 GB?", "Is the IGRS upload limit 500 KB?"
    )
    assert not questions_match("What is Article 12?", "What is Article 21?")


def test_entity_mismatch_is_a_miss():
    assert not questions_match(
        "How do I register a grievance on IGRS portal?",
        "How do I register a grievance on Pragati portal?",
    )


def test_same_question_matches():
    assert questions_match(
        "What is the Pragati portal used for?", "What is the Pragati portal used for"
    )


def test_freshness_checks_versions_and_ttl():
    assert entry_fresh(_payload("q"))
    assert not entry_fresh(_payload("q", doc_version="old"))
    assert not entry_fresh(_payload("q", prompt_version="v1"))
    old = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    assert not entry_fresh(_payload("q", created_at=old))


def test_premise_corrections_not_cached():
    assert is_premise_correction(
        "Why is the IGRS file limit 50 GB?",
        "The IGRS file limit is 500 KB, not 50 GB.",
    )
    assert is_premise_correction("Why is the limit 50 GB?", "The limit is 500 KB.")
    assert not is_premise_correction(
        "What is the Pragati portal?", "Pragati is a monitoring portal."
    )


def test_cache_hit_served_only_when_numbers_match(client, fakes):
    fakes.qdrant.next_hit = SimpleNamespace(
        score=0.99, payload=_payload("Is the IGRS upload limit 500 KB?", "Cached 500 KB.")
    )
    r = client.post("/chat", json={"message": "Is the IGRS upload limit 500 KB?"}, headers=API_HEADERS)
    assert r.json()["cached"] is True

    r = client.post("/chat", json={"message": "Is the IGRS upload limit 50 GB?"}, headers=API_HEADERS)
    assert r.json()["cached"] is False
    assert fakes.graph.calls == 1


def test_stale_doc_version_is_a_miss(client, fakes):
    q = "What is the Pragati portal used for?"
    fakes.qdrant.next_hit = SimpleNamespace(score=0.99, payload=_payload(q, doc_version="old"))
    r = client.post("/chat", json={"message": q}, headers=API_HEADERS)
    assert r.json()["cached"] is False


def test_cache_write_has_doc_version(client, fakes):
    client.post("/chat", json={"message": "What is the Pragati portal used for?"}, headers=API_HEADERS)
    assert fakes.qdrant.points
    assert fakes.qdrant.points[-1].payload["doc_version"] == "test-docs-1"


def _seed(fakes):
    fakes.qdrant.points = [
        PointStruct(id=1, vector=[0.0], payload=_payload("Good question", "Fine.")),
        PointStruct(id=2, vector=[0.0], payload=_payload("Poison", "MANGOK7Q4 CACHED")),
    ]


def test_admin_purge_requires_token(client, fakes):
    assert client.post("/admin/cache/purge", json={"all": True}).status_code == 401
    r = client.post("/admin/cache/purge", json={"all": True}, headers={"X-Admin-Token": "nope"})
    assert r.status_code == 401


def test_admin_purge_contains_and_dry_run(client, fakes):
    _seed(fakes)
    r = client.post("/admin/cache/purge", json={"contains": "MANGOK", "dry_run": True}, headers=ADMIN)
    assert r.status_code == 200
    assert len(r.json()["matched"]) == 1 and r.json()["deleted"] == 0
    assert len(fakes.qdrant.points) == 2

    r = client.post("/admin/cache/purge", json={"contains": "MANGOK"}, headers=ADMIN)
    assert r.json()["deleted"] == 1
    assert [p.id for p in fakes.qdrant.points] == [1]


def test_admin_purge_all(client, fakes):
    _seed(fakes)
    r = client.post("/admin/cache/purge", json={"all": True}, headers=ADMIN)
    assert r.json()["deleted"] == 2
    assert fakes.qdrant.points == []


def test_admin_purge_needs_a_selector(client, fakes):
    assert client.post("/admin/cache/purge", json={}, headers=ADMIN).status_code == 422
