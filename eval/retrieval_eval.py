#!/usr/bin/env python3
"""Score retrieval against the golden set using the production retrieval path.

The semantic cache is not consulted: this calls the rewriter, hybrid search,
RRF merge, and the single rerank, and never the cache lookup in main.py.
source_doc and evidence_quote are used only after retrieval, for grading.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scoring import load_items, part_found, stage_metrics  # noqa: E402

_SECRET = re.compile(
    r"(?i)(bearer\s+\S+|sk-[A-Za-z0-9]+|ya29\.\S+|AIza[0-9A-Za-z\-_]{20,}|api[_-]?key[=:]\s*\S+)"
)
_STAGE_NAMES = ("rrf_candidates", "merged", "reranked", "kept")


def describe_failure(exc: BaseException, timeout: float) -> str:
    """Short error text for one item. Secrets are stripped. Never raises."""
    if isinstance(exc, TimeoutError):
        text = f"timed out after {timeout:g}s"
    else:
        detail = (str(exc).splitlines() or [""])[0].strip()
        text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return _SECRET.sub("[redacted]", text)[:300]


def miss_fields(answerable: bool) -> dict:
    """An item that failed outright counts as a miss on every retrieval stage."""
    if not answerable:
        return {
            "recall_at_8": None,
            "stages": {},
        "dropped_by_reranker": False,
        "killed_by_threshold": False,
        "top_rerank_score": None,
        "relevant_scores": [],
    }
    zero = {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "source_hit": 0.0}
    return {
        "recall_at_8": 0.0,
        "stages": {name: dict(zero) for name in _STAGE_NAMES},
        "dropped_by_reranker": False,
        "killed_by_threshold": False,
        "top_rerank_score": None,
        "relevant_scores": [],
    }


def _chunk(hit, score) -> dict:
    payload = getattr(hit, "payload", None) or {}
    return {
        "text": payload.get("text") or "",
        "source": payload.get("document-id") or payload.get("source") or "",
        "page": payload.get("page"),
        "score": score,
    }


def _best_rr(parts, lists, docs) -> float:
    """Best reciprocal rank across the per-phrasing candidate lists."""
    from eval.scoring import reciprocal_rank

    if not lists:
        return 0.0
    return max(reciprocal_rank(parts, chunks, docs) for chunks in lists)


async def collect_pool(question: str) -> dict:
    """Rewriter, hybrid search, and RRF merge. Stops before the rerank."""
    from langchain_core.messages import HumanMessage

    from agent.lang_graph import _TOP_K, query_rewriter_node
    from app.telemetry import run_traced_in_threadpool
    from vectorDB.search import RERANK_POOL, RRF_CANDIDATES, rrf_merge, run_hybrid_search

    rewritten = await query_rewriter_node(
        {"messages": [HumanMessage(content=question)], "search_queries": []}
    )
    extras = [
        query.strip()
        for query in (rewritten.get("search_queries") or [])
        if query and query.strip().lower() != question.lower()
    ][:2]

    # Raw question, then the English and Hindi rewrites. No cache vector:
    # run_hybrid_search embeds the raw question itself, which is the same
    # embedding the cache lookup would have reused.
    phrasings = [("raw", question, None)]
    for query in extras:
        phrasings.append(("rewrite", query, None))

    batches = []
    by_phrasing = []
    for label, query, vector in phrasings:
        hits = await run_traced_in_threadpool(
            run_hybrid_search, query, RRF_CANDIDATES, vector
        )
        chunks = [_chunk(hit, None) for hit in (hits or [])]
        batches.append(hits or [])
        by_phrasing.append({"phrasing": label, "query": query, "chunks": chunks})

    merged_hits = rrf_merge(batches, limit=RERANK_POOL)
    return {
        "phrasings": [entry["query"] for entry in by_phrasing],
        "rrf_by_phrasing": by_phrasing,
        "merged": [_chunk(hit, None) for hit in merged_hits],
        "merged_hits": merged_hits,
        "top_k": _TOP_K,
        "rerank_pool": RERANK_POOL,
        "rrf_candidates": RRF_CANDIDATES,
    }


async def retrieve_item(question: str) -> dict:
    """Run one question through the same calls retriever_node makes."""
    from vectorDB.rerankers import log_skipped_threshold, reject_threshold
    from vectorDB.search import rerank_once

    pool = await collect_pool(question)
    ranked_hits = await rerank_once(question, pool["merged_hits"], pool["top_k"])
    reranked = [
        _chunk(hit, None if getattr(hit, "score", None) is None else float(hit.score))
        for hit in (ranked_hits or [])
    ]
    threshold = reject_threshold()
    if threshold is None:
        log_skipped_threshold()
        kept = list(reranked)
    else:
        kept = [chunk for chunk in reranked if float(chunk["score"] or 0) >= threshold]
    top_score = None
    if reranked:
        top_score = max(float(chunk["score"] or 0) for chunk in reranked)
    return {
        "phrasings": pool["phrasings"],
        "rrf_by_phrasing": pool["rrf_by_phrasing"],
        "merged": pool["merged"],
        "reranked": reranked,
        "kept": kept,
        "top_rerank_score": top_score,
        "top_k": pool["top_k"],
        "reject_threshold": threshold,
        "rerank_pool": pool["rerank_pool"],
        "rrf_candidates": pool["rrf_candidates"],
    }


def grade_retrieval(item: dict, retrieved: dict) -> dict:
    docs = item["source_docs"]
    parts = item["evidence_parts"]
    rrf_lists = [entry["chunks"] for entry in retrieved["rrf_by_phrasing"]]
    rrf_union: list[dict] = []
    for chunks in rrf_lists:
        rrf_union.extend(chunks)

    stages = {
        "rrf_candidates": stage_metrics(parts, rrf_union, docs),
        "merged": stage_metrics(parts, retrieved["merged"], docs),
        "reranked": stage_metrics(parts, retrieved["reranked"], docs),
        "kept": stage_metrics(parts, retrieved["kept"], docs),
    }
    if parts and docs and rrf_lists:
        stages["rrf_candidates"]["mrr"] = _best_rr(parts, rrf_lists, docs)

    in_pool = [part_found(part, retrieved["merged"], docs) for part in parts] if parts else []
    in_reranked = [part_found(part, retrieved["reranked"], docs) for part in parts] if parts else []
    in_kept = [part_found(part, retrieved["kept"], docs) for part in parts] if parts else []
    dropped_by_reranker = bool(parts) and any(
        pool and not ranked for pool, ranked in zip(in_pool, in_reranked)
    )
    killed_by_threshold = bool(parts) and any(
        ranked and not kept for ranked, kept in zip(in_reranked, in_kept)
    )
    relevant_scores = []
    for chunk in retrieved.get("reranked") or []:
        matched = bool(parts) and any(
            part_found(part, [chunk], docs) for part in parts
        )
        if matched and chunk.get("score") is not None:
            relevant_scores.append(float(chunk["score"]))
    return {
        "stages": stages,
        "dropped_by_reranker": dropped_by_reranker,
        "killed_by_threshold": killed_by_threshold,
        "recall_at_8": stages["reranked"]["recall"],
        "relevant_scores": relevant_scores,
    }


async def score_one_retrieval(index: int, total: int, item: dict, item_timeout: float) -> dict:
    """One question. Timeout and other failures become status error and do not raise."""
    row = {
        "id": item["id"],
        "language": item["language"],
        "type": item["type"],
        "difficulty": item["difficulty"],
        "answerable": item["answerable"],
        "question": item.get("question") or "",
    }
    started = time.perf_counter()
    try:
        retrieved = await asyncio.wait_for(retrieve_item(item["question"]), item_timeout)
        graded = grade_retrieval(item, retrieved) if item["answerable"] else {
            "stages": {},
            "dropped_by_reranker": False,
            "killed_by_threshold": False,
            "recall_at_8": None,
        }
        if not item["answerable"]:
            row["top_rerank_score"] = retrieved["top_rerank_score"]
            row["reranked"] = retrieved["reranked"]
        else:
            row.update(retrieved)
            row.update(graded)
        row["status"] = "ok"
        row["error"] = None
        state = "ok"
    except Exception as exc:
        row.update(miss_fields(bool(item["answerable"])))
        row["status"] = "error"
        row["error"] = describe_failure(exc, item_timeout)
        state = f"error {row['error']}"
    elapsed = time.perf_counter() - started
    print(f"retrieval {index}/{total} {item['id']} {elapsed:.1f}s {state}", flush=True)
    return row


async def run_retrieval(items: list[dict], item_timeout: float = 60.0) -> list[dict]:
    from vectorDB.search import init_services

    await asyncio.to_thread(init_services)
    rows = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        rows.append(await score_one_retrieval(index, total, item, item_timeout))
    return rows


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Retrieval eval against the golden set")
    parser.add_argument("--set", choices=("full", "sample25"), default="full")
    parser.add_argument("--item-timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    os.environ.setdefault("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    items = load_items(args.set)
    if args.limit is not None:
        items = items[: max(args.limit, 0)]
    asyncio.run(run_retrieval(items, item_timeout=args.item_timeout))


if __name__ == "__main__":
    main()
