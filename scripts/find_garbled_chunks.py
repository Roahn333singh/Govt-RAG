#!/usr/bin/env python3
"""Flag chunks that look like legacy-font / OCR garbage (low Devanagari in Hindi docs).

Usage:
  .venv/bin/python scripts/find_garbled_chunks.py
  .venv/bin/python scripts/find_garbled_chunks.py --csv /tmp/garbled.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(__file__))
from krutidev_to_unicode import looks_like_krutidev
from qdrant_client import QdrantClient

COLLECTION = "my-collection"
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Common Kruti Dev / legacy mangled markers
LEGACY_HINTS = re.compile(
    r"[àáâãäåæçèéêëìíîï]|ÿþ|[¼½¾]|[\x80-\x9f]{3,}",
    re.IGNORECASE,
)


def de_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha() or DEVANAGARI.match(c)]
    if not letters:
        return 0.0
    return sum(1 for c in letters if DEVANAGARI.match(c)) / len(letters)


def looks_garbled(text: str, doc_id: str) -> bool:
    if not text or len(text) < 40:
        return False
    if looks_like_krutidev(text):
        return True
    # Hindi-named docs with almost no Devanagari are suspicious
    hindi_name = bool(DEVANAGARI.search(doc_id or ""))
    ratio = de_ratio(text)
    if hindi_name and ratio < 0.05 and LEGACY_HINTS.search(text):
        return True
    if LEGACY_HINTS.search(text) and ratio < 0.02 and len(text) > 200:
        return True
    # High density of replacement / control-ish chars
    bad = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
    if bad > 20:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="")
    parser.add_argument("--limit-docs", type=int, default=50)
    args = parser.parse_args()

    client = QdrantClient(
        url=os.getenv("CLUSTER_ENDPOINT", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
        check_compatibility=False,
    )

    by_doc: dict[str, list] = defaultdict(list)
    offset = None
    scanned = 0
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            scanned += 1
            payload = p.payload or {}
            text = payload.get("text") or ""
            doc = payload.get("document-id") or "unknown"
            if looks_garbled(text, doc):
                by_doc[doc].append(
                    {
                        "id": p.id,
                        "page": payload.get("page"),
                        "preview": text[:120].replace("\n", " "),
                    }
                )
        if offset is None:
            break

    ranked = sorted(by_doc.items(), key=lambda kv: len(kv[1]), reverse=True)
    print(f"Scanned {scanned} chunks; {len(ranked)} docs with garbled chunks")
    for doc, rows in ranked[: args.limit_docs]:
        print(f"\n{doc}: {len(rows)} flagged")
        for r in rows[:3]:
            print(f"  p.{r['page']} {r['preview']!r}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["document_id", "point_id", "page", "preview"])
            w.writeheader()
            for doc, rows in ranked:
                for r in rows:
                    w.writerow(
                        {
                            "document_id": doc,
                            "point_id": r["id"],
                            "page": r["page"],
                            "preview": r["preview"],
                        }
                    )
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
