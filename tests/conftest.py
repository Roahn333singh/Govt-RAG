"""Shared fakes: no OpenRouter, no Qdrant, no model downloads."""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

os.environ.update(
    {
        "OPENROUTER_API_KEY": "test-openrouter",
        "CLUSTER_ENDPOINT": "http://qdrant.invalid:6333",
        "QDRANT_API_KEY": "",
        "APP_API_KEY": "test-api-key",
        "SESSION_SIGNING_KEY": "test-signing-key",
        "ADMIN_TOKEN": "test-admin-token",
        "ALLOWED_ORIGINS": "https://manual-ai-rag-nszha5soya-ey.a.run.app,http://testserver",
        "RATE_LIMIT_STORAGE_URI": "memory://",
        "GIT_SHA": "testsha",
        "DOC_VERSION": "test-docs-1",
    }
)

API_HEADERS = {"X-API-Key": "test-api-key"}


class FakeEmbedder:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        # Deterministic, length-1536, roughly distinct per text
        seed = sum(ord(c) for c in text) % 997
        return [((seed + i) % 13) / 13.0 for i in range(1536)]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self._vec(text)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


class FakeQdrant:
    """Stores cache points in memory; query returns a configurable hit."""

    def __init__(self):
        self.points: list = []
        self.next_hit = None  # SimpleNamespace(score=..., payload=...)
        self.upserts = 0

    def query_points(self, collection_name=None, **kwargs):
        hits = [self.next_hit] if self.next_hit else []
        return SimpleNamespace(points=hits)

    def upsert(self, collection_name=None, points=None, **kwargs):
        self.upserts += 1
        self.points.extend(points or [])

    def get_collection(self, *a, **k):
        return {}

    def collection_exists(self, *a, **k):
        return True

    def scroll(self, *a, **k):
        return [], None

    def delete(self, *a, **k):
        return None


class FakeGraph:
    def __init__(self, answer="Answer text. Sources: Doc.pdf, Page 1", sources=None, delay=0.0):
        self.answer = answer
        self.sources = sources if sources is not None else [
            {"source_document": "Doc.pdf", "page_start": 1, "content": "x", "score": 1.0}
        ]
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, state, config=None):
        import asyncio

        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"messages": [AIMessage(content=self.answer)], "sources": self.sources}

    async def astream_events(self, state, config=None, version="v2"):
        from langchain_core.messages import AIMessageChunk

        self.calls += 1
        words = self.answer.split(" ")
        for i, w in enumerate(words):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "generator"},
                "data": {"chunk": AIMessageChunk(content=w + (" " if i < len(words) - 1 else ""))},
            }
        yield {
            "event": "on_chain_end",
            "name": "generator",
            "data": {"output": {"sources": self.sources}},
        }


@pytest.fixture
def fakes(monkeypatch):
    import main
    from vectorDB import search

    emb = FakeEmbedder()
    qd = FakeQdrant()
    graph = FakeGraph()
    monkeypatch.setattr(search, "dense_embedder", emb)
    monkeypatch.setattr(search, "qdrant_client", qd)
    monkeypatch.setattr(search, "sparse_embedder", object())
    monkeypatch.setattr(search, "reranker", object())
    monkeypatch.setattr(search, "_services_ready", True)
    monkeypatch.setattr(main, "get_compile_all", lambda: (lambda: graph))
    main.limiter.reset()
    if hasattr(main, "_exact_cache"):
        main._exact_cache.clear()
    return SimpleNamespace(embedder=emb, qdrant=qd, graph=graph, main=main)


@pytest.fixture
def client(fakes):
    from fastapi.testclient import TestClient

    return TestClient(fakes.main.app)
