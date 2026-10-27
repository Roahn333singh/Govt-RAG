from agent.guards import INJECTION_REFUSAL, cited_sources
from tests.conftest import API_HEADERS

DOCS = [
    {"source_document": "IGRSFAQ.pdf", "page_start": 2, "content": "a", "score": 3},
    {"source_document": "Pragati.pdf", "page_start": 5, "content": "b", "score": 2},
    {"source_document": "FMISC.pdf", "page_start": 1, "content": "c", "score": 1},
    {"source_document": "GPF.pdf", "page_start": 9, "content": "d", "score": 0},
]


def test_only_cited_documents_are_returned():
    answer = "The limit is 500 KB.\n\nSources: IGRSFAQ.pdf, Page: 2"
    assert [s["source_document"] for s in cited_sources(answer, DOCS)] == ["IGRSFAQ.pdf"]


def test_missing_sources_line_falls_back_to_top_three():
    assert len(cited_sources("The limit is 500 KB.", DOCS)) == 3


def test_refusals_have_no_sources():
    assert cited_sources(INJECTION_REFUSAL, DOCS) == []
    assert cited_sources(
        "The provided documents do not cover this topic.", DOCS
    ) == []


def test_chat_filters_uncited_sources(client, fakes):
    fakes.graph.answer = "Pragati is a monitoring portal.\n\nSources: Pragati.pdf, Page: 5"
    fakes.graph.sources = DOCS
    r = client.post("/chat", json={"message": "What is the Pragati portal?"}, headers=API_HEADERS)
    body = r.json()
    assert [s["source_document"] for s in body["sources"]] == ["Pragati.pdf"]


def test_chat_refusal_returns_empty_sources(client, fakes):
    fakes.graph.answer = "I am an AI assistant specialized in UP Government Data. I cannot answer queries about unrelated topics."
    fakes.graph.sources = DOCS
    r = client.post("/chat", json={"message": "What is the capital of France?"}, headers=API_HEADERS)
    assert r.json()["sources"] == []
