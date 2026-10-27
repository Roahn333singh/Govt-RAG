#!/usr/bin/env python3
"""Delete all points for documents flagged by find_garbled_chunks.py.

After deletion, re-ingest those PDFs with OCR (hin+eng) via the existing
vectorize scripts, then run remove_duplicates.py.

Usage:
  .venv/bin/python scripts/find_garbled_chunks.py --csv /tmp/garbled.csv
  .venv/bin/python scripts/delete_garbled_docs.py --csv /tmp/garbled.csv --dry-run
  .venv/bin/python scripts/delete_garbled_docs.py --csv /tmp/garbled.csv --min-chunks 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(override=True)

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

COLLECTION = "my-collection"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--min-chunks", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["document_id"]] += 1

    targets = sorted(
        (d for d, n in counts.items() if n >= args.min_chunks),
        key=lambda d: -counts[d],
    )
    print(f"Docs to clear (>= {args.min_chunks} flagged): {len(targets)}")
    for d in targets:
        print(f"  {counts[d]:4d}  {d}")

    if args.dry_run or not targets:
        print("Dry-run / nothing to do")
        return

    client = QdrantClient(
        url=os.getenv("CLUSTER_ENDPOINT", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=120,
        check_compatibility=False,
    )
    for doc in targets:
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="document-id", match=MatchValue(value=doc))]
            ),
        )
        print(f"deleted points for {doc}")
    print("Done. Re-OCR and re-ingest these files next.")


if __name__ == "__main__":
    main()
