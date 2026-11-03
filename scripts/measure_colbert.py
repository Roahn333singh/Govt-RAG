#!/usr/bin/env python3
"""Measure ColBERT query-encode and MaxSim rerank latency on a warm process.

Does not modify collections. Requires COLBERT_COLLECTION and a few point ids
(or scrolls them from the dest collection).
"""

from __future__ import annotations

import argparse
import os
import resource
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)


def _rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_ = usage if sys.platform == "darwin" else usage * 1024
    return bytes_ / 1024 / 1024


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--query", default="What are the financial powers of a Chief Engineer?")
    args = parser.parse_args()

    os.environ.setdefault("RERANKER", "colbert")
    from vectorDB import search
    from vectorDB.rerankers import load_reranker

    search.init_services()
    backend = load_reranker("colbert")

    # Warm encode.
    backend.encode_query(args.query)

    points, _ = search.qdrant_client.scroll(
        collection_name=backend.collection,
        limit=args.candidates,
        with_payload=True,
        with_vectors=False,
    )
    if len(points) < 2:
        raise SystemExit(f"need candidates in {backend.collection}")

    encode_ms = []
    rerank_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        backend.encode_query(args.query)
        encode_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        ranked = backend.rerank(args.query, points, min(8, len(points)))
        rerank_ms.append((time.perf_counter() - t1) * 1000)
        if not ranked:
            raise SystemExit("colbert rerank returned no hits")

    print(
        f"query_encode_ms p50={percentile(encode_ms, 0.5):.1f} "
        f"p95={percentile(encode_ms, 0.95):.1f} mean={statistics.mean(encode_ms):.1f}"
    )
    print(
        f"rerank_ms p50={percentile(rerank_ms, 0.5):.1f} "
        f"p95={percentile(rerank_ms, 0.95):.1f} mean={statistics.mean(rerank_ms):.1f}"
    )
    print(f"peak_rss_mib={_rss_mib():.1f} candidates={len(points)} runs={args.runs}")


if __name__ == "__main__":
    main()
