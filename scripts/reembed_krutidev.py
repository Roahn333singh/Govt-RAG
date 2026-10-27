#!/usr/bin/env python3
"""Convert Kruti Dev chunks to Unicode and re-embed them.

Only writes a point when the conversion actually adds Devanagari, so a bad
match is left untouched. Defaults to the Disciplinary Proceeding Rule PPT.

Usage:
  .venv/bin/python scripts/reembed_krutidev.py --dry-run
  .venv/bin/python scripts/reembed_krutidev.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(override=True)

from qdrant_client.models import PointStruct, SparseVector

from krutidev_to_unicode import kru2uni, looks_like_krutidev
from patch_point_text import COLLECTION, DENSE_NAME, SPARSE_NAME, get_client

DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def _dev_count(text: str) -> int:
    return len(DEVANAGARI.findall(text or ""))


def improved(original: str, converted: str) -> bool:
    return _dev_count(converted) >= _dev_count(original) + 12 and converted != original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--contains", default="Disciplinary Proceeding")
    args = parser.parse_args()

    client = get_client()
    from vectorDB.embeddings import get_dense_embedder
    from fastembed import SparseTextEmbedding

    dense = None if args.dry_run else get_dense_embedder()
    sparse = None if args.dry_run else SparseTextEmbedding(model_name="Qdrant/bm25")

    offset = None
    changed = 0
    scanned = 0
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = dict(point.payload or {})
            doc = payload.get("document-id") or ""
            text = payload.get("text") or ""
            if args.contains.lower() not in doc.lower():
                continue
            scanned += 1
            if not looks_like_krutidev(text):
                continue
            converted = kru2uni(text)
            if not improved(text, converted):
                print(f"skip {point.id} page={payload.get('page')} (conversion did not add Devanagari)")
                continue
            changed += 1
            print(f"{'would update' if args.dry_run else 'update'} {point.id} page={payload.get('page')}")
            print("   ", converted[:160].replace("\n", " "))
            if args.dry_run:
                continue
            payload["text"] = converted
            dense_vec = dense.embed_query(converted)
            sp = list(sparse.embed([converted]))[0]
            client.upsert(
                collection_name=COLLECTION,
                points=[
                    PointStruct(
                        id=point.id,
                        payload=payload,
                        vector={
                            DENSE_NAME: dense_vec,
                            SPARSE_NAME: SparseVector(indices=list(sp.indices), values=list(sp.values)),
                        },
                    )
                ],
            )
        if offset is None:
            break
    print(f"{'would change' if args.dry_run else 'changed'} {changed} / scanned {scanned}")


if __name__ == "__main__":
    main()
