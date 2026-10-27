#!/usr/bin/env python3
"""Retrieval-only reranker score report, used to tune the reject threshold.

No LLM calls; each question costs one embedding request.

Usage:
  .venv/bin/python scripts/rerank_scores.py
  .venv/bin/python scripts/rerank_scores.py --model BAAI/bge-reranker-base --limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"), override=True)

OFF_DOMAIN = [
    "What is the capital of France?",
    "Give me a recipe for chicken biryani.",
    "Who won the 2018 FIFA World Cup?",
    "How do I reverse a linked list in Python?",
    "What is the boiling point of mercury?",
    "Recommend a good laptop under 50000 rupees.",
    "भारत का राष्ट्रीय पशु कौन सा है?",
    "Write a poem about the monsoon.",
    "What is the price of Bitcoin today?",
    "How many moons does Jupiter have?",
]


def load_questions(limit: int) -> list[str]:
    gold = os.path.join(ROOT, "eval", "gold.json")
    if os.path.exists(gold):
        items = json.load(open(gold, encoding="utf-8"))
        qs = [i["question"] for i in items if not i.get("must_refuse")]
    else:
        txt = ""
        for name in ("evaluation_qa.md", "nivida_evaluation_qa.md"):
            path = os.path.join(ROOT, name)
            if os.path.exists(path):
                txt += open(path, encoding="utf-8").read()
        found = re.findall(r"Question:\*\*\s*(.+)|^### Q\d+\.\s*(.+)$", txt, re.M)
        qs = [a or b for a, b in found]
    return qs[:limit]


def summarize(label: str, scores: list[float]) -> None:
    if not scores:
        return
    s = sorted(scores)
    pct = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    print(
        f"{label:<12} n={len(s):<4} min={s[0]:7.2f} p10={pct(0.1):7.2f} "
        f"median={statistics.median(s):7.2f} p90={pct(0.9):7.2f} max={s[-1]:7.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("RERANKER_MODEL"))
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    if args.model:
        os.environ["RERANKER_MODEL"] = args.model

    from vectorDB import search

    search.init_services()
    print(f"Reranker: {os.getenv('RERANKER_MODEL') or search.DEFAULT_RERANKER}\n")

    in_top, off_top, latencies = [], [], []
    for label, questions, bucket in (
        ("in-domain", load_questions(args.limit), in_top),
        ("off-domain", OFF_DOMAIN, off_top),
    ):
        for q in questions:
            t0 = time.perf_counter()
            hits = search.run_hybrid_search(q, top_k=8)
            latencies.append(time.perf_counter() - t0)
            top = hits[0].score if hits else float("-inf")
            bucket.append(top)
            doc = (hits[0].payload.get("document-id") if hits else "-") or "-"
            print(f"  [{label}] {top:7.2f}  {q[:70]!r}  -> {doc[:40]}")

    print()
    summarize("in-domain", in_top)
    summarize("off-domain", off_top)
    summarize("latency s", latencies)
    if in_top and off_top:
        best = max(off_top)
        missed = sum(1 for s in in_top if s <= best)
        print(
            f"\nA threshold just above the best off-domain score ({best:.2f}) "
            f"would refuse {missed}/{len(in_top)} in-domain questions."
        )


if __name__ == "__main__":
    main()
