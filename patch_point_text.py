"""
Surgically update one (or more) Qdrant points without re-ingesting a whole PDF.

Examples:
  # Preview only (no write)
  uv run python patch_point_text.py \\
    --point-id 31dbd919-51c9-4fd9-8eca-374dfad7a78d \\
    --old "www.pragati.upsdc.gov.in" \\
    --new "https://example.com/pragati-new" \\
    --dry-run

  # Apply + clear semantic cache
  uv run python patch_point_text.py \\
    --point-id 31dbd919-51c9-4fd9-8eca-374dfad7a78d \\
    --point-id 63289c54-8810-4bdf-942d-c5d8d8d8dd50 \\
    --old "www.pragati.upsdc.gov.in" \\
    --new "https://example.com/pragati-new" \\
    --clear-cache

  # Find every point containing old text, then patch
  uv run python patch_point_text.py \\
    --find-old "www.pragati.upsdc.gov.in" \\
    --old "www.pragati.upsdc.gov.in" \\
    --new "https://example.com/pragati-new" \\
    --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    VectorParams,
)

load_dotenv(override=True)

COLLECTION = "my-collection"
CACHE = "semantic-query-cache"
DENSE_NAME = "abstract-dense-vector"
SPARSE_NAME = "sparse-vector"


def get_client() -> QdrantClient:
    url = os.getenv("CLUSTER_ENDPOINT", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(
        url=url, api_key=api_key, timeout=60.0, check_compatibility=False
    )


def find_points_with_text(client: QdrantClient, needle: str) -> list[str]:
    ids: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            text = (p.payload or {}).get("text") or ""
            if needle in text:
                ids.append(str(p.id))
                doc = (p.payload or {}).get("document-id")
                page = (p.payload or {}).get("page")
                print(f"  found {p.id}  doc={doc}  page={page}")
        if offset is None:
            break
    return ids


def clear_cache(client: QdrantClient) -> None:
    if client.collection_exists(CACHE):
        client.delete_collection(CACHE)
    client.create_collection(
        CACHE, vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    client.create_payload_index(
        CACHE, field_name="language", field_schema=PayloadSchemaType.KEYWORD
    )
    print(f"✅ Cleared and recreated '{CACHE}'")


def patch_one(
    client: QdrantClient,
    dense_embedder,
    sparse_embedder,
    point_id: str,
    old: str,
    new: str,
    dry_run: bool,
) -> bool:
    records = client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    if not records:
        print(f"❌ Point not found: {point_id}")
        return False

    point = records[0]
    payload = dict(point.payload or {})
    text = payload.get("text") or ""
    if old not in text:
        print(f"⚠️  '{old}' not found in {point_id} — skip")
        print(f"    doc={payload.get('document-id')} page={payload.get('page')}")
        return False

    new_text = text.replace(old, new)
    count = text.count(old)
    print(f"\n── {point_id}")
    print(f"   document-id : {payload.get('document-id')}")
    print(f"   source      : {payload.get('source')}")
    print(f"   page        : {payload.get('page')}")
    print(f"   replacements: {count}")
    print(f"   old snippet : …{old}…")
    print(f"   new snippet : …{new}…")

    if dry_run:
        print("   DRY-RUN — not written")
        return True

    payload["text"] = new_text
    dense = dense_embedder.embed_query(new_text)
    if hasattr(dense_embedder, "embed_documents"):
        dense = dense_embedder.embed_documents([new_text])[0]
    sparse = list(sparse_embedder.embed([new_text]))[0]

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point.id,
                payload=payload,
                vector={
                    DENSE_NAME: dense,
                    SPARSE_NAME: SparseVector(
                        indices=list(sparse.indices),
                        values=list(sparse.values),
                    ),
                },
            )
        ],
    )
    print("   ✅ upserted (text + dense + sparse)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch Qdrant point text and re-embed without full PDF reingest"
    )
    parser.add_argument(
        "--point-id",
        action="append",
        default=[],
        help="Point UUID to patch (repeatable)",
    )
    parser.add_argument(
        "--find-old",
        help="Scan collection for this substring and use matching point ids",
    )
    parser.add_argument("--old", required=True, help="Exact old string to replace")
    parser.add_argument("--new", required=True, help="Replacement string")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without writing"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Wipe semantic-query-cache after a successful write",
    )
    args = parser.parse_args()

    if args.old == args.new:
        print("❌ --old and --new are identical")
        sys.exit(1)

    client = get_client()
    point_ids = list(args.point_id)

    if args.find_old:
        print(f"🔍 Scanning for {args.find_old!r}…")
        found = find_points_with_text(client, args.find_old)
        print(f"   {len(found)} point(s)")
        point_ids = list(dict.fromkeys(point_ids + found))

    if not point_ids:
        print("❌ No point ids. Pass --point-id and/or --find-old")
        sys.exit(1)

    dense_embedder = sparse_embedder = None
    if not args.dry_run:
        from vectorDB import search

        search.init_services()
        dense_embedder = search.dense_embedder
        sparse_embedder = search.sparse_embedder

    ok = 0
    for pid in point_ids:
        if patch_one(
            client, dense_embedder, sparse_embedder, pid, args.old, args.new, args.dry_run
        ):
            ok += 1

    print(f"\nDone: {ok}/{len(point_ids)} point(s) {'previewed' if args.dry_run else 'patched'}")

    if args.clear_cache and not args.dry_run and ok:
        clear_cache(client)
    elif args.clear_cache and args.dry_run:
        print("(dry-run: cache not cleared)")


if __name__ == "__main__":
    main()
