from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent.guards import parse_bilingual_rewrite


def _hit(doc, page, score, text="relevant text"):
    return SimpleNamespace(
        score=score,
        payload={"document-id": doc, "page": page, "text": text},
    )


def test_rrf_merge_dedupes_and_caps_the_pool():
    from vectorDB.search import rrf_merge

    first = [_hit(f"a{i}.pdf", i, 0) for i in range(15)]
    second = [_hit("a0.pdf", 0, 0)] + [_hit(f"b{i}.pdf", i, 0) for i in range(15)]
    merged = rrf_merge([first, second], limit=20)
    ids = [hit.payload["document-id"] for hit in merged]
    assert len(merged) == 20
    assert ids.count("a0.pdf") == 1
    assert ids[0] == "a0.pdf"


def test_truncate_bounds_unscored_text():
    from vectorDB.search import RERANK_MAX_TOKENS, truncate_for_rerank

    text = "word " * 2000
    assert len(truncate_for_rerank(text)) <= RERANK_MAX_TOKENS * 4


async def test_rerank_semaphore_allows_one(monkeypatch):
    import asyncio
    import threading
    import time

    from vectorDB import search

    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    def slow(question, hits, top_k):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return hits[:top_k]

    monkeypatch.setattr(search, "rerank_hits", slow)
    await asyncio.gather(
        search.rerank_once("q", [_hit("a.pdf", 1, 1.0)], 1),
        search.rerank_once("q", [_hit("b.pdf", 1, 1.0)], 1),
    )
    assert state["peak"] == 1


def test_bilingual_rewrite_keeps_both_lines():
    raw = "EN: IGRS file size upload 500 KB\nHI: आईजीआरएस फाइल साइज 500 केबी"
    assert parse_bilingual_rewrite(raw, "what is the limit") == [
        "IGRS file size upload 500 KB",
        "आईजीआरएस फाइल साइज 500 केबी",
    ]


def test_bilingual_rewrite_drops_echo_of_the_question():
    assert parse_bilingual_rewrite("EN: same question\nHI:", "same question") == []


async def test_retriever_reranks_the_fused_pool_once(monkeypatch):
    from agent import lang_graph
    from vectorDB.search import RERANK_POOL, RRF_CANDIDATES

    calls = []
    reranks = []

    def fake_search(question, limit, vector):
        calls.append((question, limit, vector))
        return [_hit(f"{question[:2]}-{i}.pdf", i, 2.0 - i * 0.1) for i in range(10)]

    async def fake_rerank(question, hits, top_k):
        reranks.append((question, len(hits), top_k))
        ranked = sorted(hits, key=lambda h: float(h.score or 0), reverse=True)
        return ranked[:top_k]

    monkeypatch.setattr(lang_graph, "run_hybrid_search", fake_search)
    monkeypatch.setattr(lang_graph, "rerank_once", fake_rerank)
    out = await lang_graph.retriever_node(
        {
            "messages": [HumanMessage(content="raw question")],
            "search_queries": ["english query", "हिंदी प्रश्न"],
            "query_vector": [0.1],
        }
    )
    assert [c[0] for c in calls] == ["raw question", "english query", "हिंदी प्रश्न"]
    assert calls[0][2] == [0.1]
    assert all(c[1] == RRF_CANDIDATES for c in calls)
    assert len(reranks) == 1
    assert reranks[0][0] == "raw question"
    assert reranks[0][1] <= RERANK_POOL
    assert reranks[0][2] == 8
    assert out["context"] != "REJECT"
    assert len(out["sources"]) == 8


async def test_one_passing_chunk_is_enough(monkeypatch):
    from agent import lang_graph

    def fake_search(question, limit, vector):
        if question == "raw":
            return [_hit("weak.pdf", 1, -50.0)]
        return [_hit("good.pdf", 2, -1.0)]

    async def fake_rerank(question, hits, top_k):
        return sorted(hits, key=lambda h: float(h.score or 0), reverse=True)[:top_k]

    monkeypatch.setattr(lang_graph, "run_hybrid_search", fake_search)
    monkeypatch.setattr(lang_graph, "rerank_once", fake_rerank)
    out = await lang_graph.retriever_node(
        {"messages": [HumanMessage(content="raw")], "search_queries": ["en"], "query_vector": None}
    )
    assert out["context"] != "REJECT"
    assert out["sources"][0]["source_document"] == "good.pdf"


async def test_all_below_threshold_refuses(monkeypatch):
    from agent import lang_graph

    async def fake_rerank(question, hits, top_k):
        return hits[:top_k]

    monkeypatch.setattr(
        lang_graph,
        "run_hybrid_search",
        lambda *a, **k: [_hit("noise.pdf", 1, -99.0)],
    )
    monkeypatch.setattr(lang_graph, "rerank_once", fake_rerank)
    out = await lang_graph.retriever_node(
        {"messages": [HumanMessage(content="raw")], "search_queries": [], "query_vector": None}
    )
    assert out == {"context": "REJECT", "sources": []}
