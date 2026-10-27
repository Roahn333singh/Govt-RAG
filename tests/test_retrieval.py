from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent.guards import parse_bilingual_rewrite


def _hit(doc, page, score, text="relevant text"):
    return SimpleNamespace(
        score=score,
        payload={"document-id": doc, "page": page, "text": text},
    )


def test_bilingual_rewrite_keeps_both_lines():
    raw = "EN: IGRS file size upload 500 KB\nHI: आईजीआरएस फाइल साइज 500 केबी"
    assert parse_bilingual_rewrite(raw, "what is the limit") == [
        "IGRS file size upload 500 KB",
        "आईजीआरएस फाइल साइज 500 केबी",
    ]


def test_bilingual_rewrite_drops_echo_of_the_question():
    assert parse_bilingual_rewrite("EN: same question\nHI:", "same question") == []


async def test_retriever_keeps_top_8_above_threshold(monkeypatch):
    from agent import lang_graph

    calls = []

    def fake_search(question, top_k, vector):
        calls.append((question, top_k, vector))
        return [_hit(f"{question[:2]}-{i}.pdf", i, 2.0 - i * 0.1) for i in range(10)]

    monkeypatch.setattr(lang_graph, "run_hybrid_search", fake_search)
    out = await lang_graph.retriever_node(
        {
            "messages": [HumanMessage(content="raw question")],
            "search_queries": ["english query", "हिंदी प्रश्न"],
            "query_vector": [0.1],
        }
    )
    assert [c[0] for c in calls] == ["raw question", "english query", "हिंदी प्रश्न"]
    assert calls[0][2] == [0.1]
    assert all(c[1] == 8 for c in calls)
    assert out["context"] != "REJECT"
    assert len(out["sources"]) == 8


async def test_one_passing_chunk_is_enough(monkeypatch):
    from agent import lang_graph

    def fake_search(question, top_k, vector):
        if question == "raw":
            return [_hit("weak.pdf", 1, -50.0)]
        return [_hit("good.pdf", 2, -1.0)]

    monkeypatch.setattr(lang_graph, "run_hybrid_search", fake_search)
    out = await lang_graph.retriever_node(
        {"messages": [HumanMessage(content="raw")], "search_queries": ["en"], "query_vector": None}
    )
    assert out["context"] != "REJECT"
    assert out["sources"][0]["source_document"] == "good.pdf"


async def test_all_below_threshold_refuses(monkeypatch):
    from agent import lang_graph

    monkeypatch.setattr(
        lang_graph,
        "run_hybrid_search",
        lambda *a, **k: [_hit("noise.pdf", 1, -99.0)],
    )
    out = await lang_graph.retriever_node(
        {"messages": [HumanMessage(content="raw")], "search_queries": [], "query_vector": None}
    )
    assert out == {"context": "REJECT", "sources": []}
