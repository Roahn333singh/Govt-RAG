#!/usr/bin/env python3
"""Time each reranker on 20 documents of 512 tokens.

Each backend runs in its own process: 30 sequential passes, 2 intra-op threads.
Prints p50, p95, and peak RSS. Flags peak RSS above 6 GiB or p95 above 5 s.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKENDS = ("minilm", "mmarco", "gte", "none")
RSS_LIMIT = 6 * (1024**3)
P95_LIMIT_S = 5.0
RUNS = 30
DOCS = 20
TOKENS = 512


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak) * 1024


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def bench_one(name: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["ONNXRUNTIME_INTRA_OP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["RERANKER"] = name

    from types import SimpleNamespace

    from vectorDB.rerankers import load_reranker

    backend = load_reranker(name)
    text = "token " * TOKENS
    docs = [
        SimpleNamespace(payload={"text": text}, score=1.0 / (index + 1), rrf_score=1.0 / (60 + index + 1))
        for index in range(DOCS)
    ]
    backend.rerank("bench query", docs, 8)
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        backend.rerank("bench query", docs, 8)
        samples.append(time.perf_counter() - started)
    p50 = _percentile(samples, 0.50)
    p95 = _percentile(samples, 0.95)
    rss = _rss_bytes()
    print(f"RESULT {name} p50_s={p50:.4f} p95_s={p95:.4f} rss_bytes={rss}")


def _parse(line: str) -> dict:
    parts = line.split()
    fields = {"name": parts[1]}
    for part in parts[2:]:
        key, value = part.split("=", 1)
        fields[key] = float(value)
    return fields


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        bench_one(sys.argv[2])
        return 0
    flagged = False
    print(f"bench {DOCS} docs x {TOKENS} tokens, {RUNS} runs, 2 threads")
    for name in BACKENDS:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--only", name],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            print(f"FLAG {name} failed to load or run (exit {completed.returncode})")
            for line in detail[-12:]:
                print(line)
            flagged = True
            continue
        result = None
        for line in (completed.stdout or "").splitlines():
            if line.startswith("RESULT "):
                result = _parse(line)
        if result is None:
            print(f"FLAG {name} produced no RESULT line")
            flagged = True
            continue
        gib = result["rss_bytes"] / (1024**3)
        print(
            f"{name:<8} p50={result['p50_s']:.3f}s  p95={result['p95_s']:.3f}s  "
            f"peak RSS={gib:.2f} GiB"
        )
        if result["rss_bytes"] > RSS_LIMIT:
            print(f"FLAG {name} peak RSS {gib:.2f} GiB exceeds 6 GiB")
            flagged = True
        if result["p95_s"] > P95_LIMIT_S:
            print(f"FLAG {name} p95 {result['p95_s']:.3f}s exceeds 5 s")
            flagged = True
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
