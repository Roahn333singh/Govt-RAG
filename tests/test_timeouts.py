import asyncio

from langchain_core.messages import HumanMessage

from tests.conftest import API_HEADERS


def test_slow_pipeline_returns_clean_503(fakes, client, monkeypatch):
    monkeypatch.setattr(fakes.main, "REQUEST_TIMEOUT_SECONDS", 0.3)
    fakes.graph.delay = 2.0
    r = client.post("/chat", json={"message": "Slow question about IGRS"}, headers=API_HEADERS)
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert r.json()["detail"] == "Service busy, please retry"
    assert "Traceback" not in r.text


def test_pipeline_exception_returns_503_without_trace(fakes, client):
    async def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    fakes.graph.ainvoke = boom
    r = client.post("/chat", json={"message": "Question that explodes"}, headers=API_HEADERS)
    assert r.status_code == 503
    assert "secret internal detail" not in r.text


async def test_rewriter_falls_back_to_raw_question(monkeypatch):
    from agent import lang_graph

    class SlowLLM:
        async def ainvoke(self, prompt):
            await asyncio.sleep(5)

    monkeypatch.setattr(lang_graph, "get_llm", lambda temperature=0.7: SlowLLM())
    monkeypatch.setattr(lang_graph, "REWRITER_TIMEOUT_SECONDS", 0.1)
    out = await lang_graph.query_rewriter_node(
        {"messages": [HumanMessage(content="What is the IGRS file size limit?")]}
    )
    assert out["search_query"] == "What is the IGRS file size limit?"
