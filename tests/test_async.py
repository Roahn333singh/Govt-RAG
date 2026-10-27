"""Blocking embedder / Qdrant calls must not stall the event loop."""

import ast
import asyncio
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import API_HEADERS

ROOT = Path(__file__).resolve().parents[1]

# Sync methods that hit the network or burn CPU; inside async code they must be
# passed to to_thread / run_in_threadpool, never called directly.
BLOCKING = {"embed_query", "embed_documents", "query_points", "upsert", "rerank", "run_hybrid_search", "get_collection"}


def _direct_blocking_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in BLOCKING:
                    bad.append(f"{path.name}:{node.lineno} {fn.name} -> {name}()")
    return bad


@pytest.mark.parametrize("rel", ["main.py", "agent/lang_graph.py"])
def test_no_direct_blocking_calls_in_async_code(rel):
    assert _direct_blocking_calls(ROOT / rel) == []


async def test_parallel_requests_overlap_and_health_stays_fast(fakes):
    fakes.embedder.delay = 2.0
    loop = asyncio.get_running_loop()
    from concurrent.futures import ThreadPoolExecutor

    loop.set_default_executor(ThreadPoolExecutor(max_workers=16))
    transport = httpx.ASGITransport(app=fakes.main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        t0 = time.perf_counter()
        chats = [
            ac.post("/chat", json={"message": f"IGRS file size question {i}"}, headers=API_HEADERS)
            for i in range(5)
        ]
        chat_tasks = [asyncio.create_task(c) for c in chats]
        await asyncio.sleep(0.3)
        h0 = time.perf_counter()
        health = await ac.get("/health")
        health_ms = (time.perf_counter() - h0) * 1000
        results = await asyncio.gather(*chat_tasks)
        elapsed = time.perf_counter() - t0

    assert health.status_code == 200
    assert health_ms < 500
    assert all(r.status_code == 200 for r in results)
    assert elapsed < 4.0, f"5 x 2s embeds took {elapsed:.1f}s — calls are serialised"
