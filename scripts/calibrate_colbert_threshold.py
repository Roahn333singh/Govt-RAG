#!/usr/bin/env python3
"""Print MaxSim score distributions and a suggested COLBERT_REJECT_THRESHOLD.

Runs the in-process retrieval eval (prod50 or sample25) with RERANKER=colbert.
Does not apply the threshold. Set COLBERT_COLLECTION first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("prod50", "sample25", "full"), default="prod50")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.environ["RERANKER"] = "colbert"
    if not (os.getenv("COLBERT_COLLECTION") or "").strip():
        raise SystemExit("COLBERT_COLLECTION is not set")

    from eval.retrieval_eval import run_retrieval
    from eval.run_eval import apply_limit, retrieval_summary
    from eval.scoring import load_items
    from vectorDB.rerankers import suggest_threshold

    items = apply_limit(load_items(args.set), args.limit)
    rows = await run_retrieval(items)
    summary = retrieval_summary(rows)
    print(json.dumps(summary.get("top_score"), indent=2))

    records = [
        {
            "answerable": bool(row.get("answerable")),
            "top_score": row.get("top_rerank_score"),
            "relevant_scores": row.get("relevant_scores")
            or (
                [row["top_rerank_score"]]
                if row.get("answerable") and row.get("top_rerank_score") is not None
                else []
            ),
        }
        for row in rows
        if row.get("status") == "ok"
    ]
    suggested = suggest_threshold(records)
    print(f"suggested COLBERT_REJECT_THRESHOLD={suggested}")
    print("Not applied. Export it before serving if you want the floor.")


if __name__ == "__main__":
    asyncio.run(main())
