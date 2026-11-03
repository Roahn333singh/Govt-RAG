#!/usr/bin/env python3
"""Build a ColBERT-augmented copy of the live Qdrant collection.

Copies payloads, dense vectors, and sparse vectors from SOURCE into a NEW
collection, then adds a named multivector "colbert" (jinaai/jina-colbert-v2,
MAX_SIM, int8 scalar quantization, HNSW m=0). Never writes to the live
collection.

Usage:
  uv run python scripts/index_colbert.py
  uv run python scripts/index_colbert.py --source my-collection --dest my-collection-colbert
  uv run python scripts/index_colbert.py --resume   # skip points that already have colbert
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HasVectorCondition,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from vectorDB.rerankers import COLBERT_DIM, COLBERT_MODEL, COLBERT_VECTOR_NAME  # noqa: E402

DENSE_NAME = "abstract-dense-vector"
SPARSE_NAME = "sparse-vector"
DEFAULT_SOURCE = "my-collection"
DEFAULT_DEST = "my-collection-colbert"


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _as_list(matrix) -> list:
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return [row.tolist() if hasattr(row, "tolist") else list(row) for row in matrix]


def _sparse_payload(vec) -> SparseVector | None:
    if vec is None:
        return None
    indices = getattr(vec, "indices", None)
    values = getattr(vec, "values", None)
    if indices is None or values is None:
        return None
    return SparseVector(
        indices=list(indices),
        values=[float(value) for value in values],
    )


def ensure_dest(client: QdrantClient, source: str, dest: str) -> None:
    if client.collection_exists(dest):
        print(f"dest collection exists: {dest}")
        return
    if not client.collection_exists(source):
        raise SystemExit(f"source collection missing: {source}")
    print(f"creating {dest} from {source} schema + colbert multivector...")
    client.create_collection(
        collection_name=dest,
        vectors_config={
            DENSE_NAME: VectorParams(size=1536, distance=Distance.COSINE),
            COLBERT_VECTOR_NAME: VectorParams(
                size=COLBERT_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=HnswConfigDiff(m=0),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            ),
        },
        sparse_vectors_config={SPARSE_NAME: SparseVectorParams()},
    )
    # Mirror useful payload indexes from the live collection.
    for field in ("document-id", "language", "source"):
        try:
            from qdrant_client.models import PayloadSchemaType

            client.create_payload_index(
                collection_name=dest,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            print(f"  payload index {field}: {type(exc).__name__}")


def already_indexed(client: QdrantClient, dest: str) -> set:
    done: set = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=dest,
            scroll_filter=Filter(
                must=[HasVectorCondition(has_vector=COLBERT_VECTOR_NAME)]
            ),
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points or []:
            done.add(point.id)
        if not points or offset is None:
            break
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.getenv("COLBERT_SOURCE", DEFAULT_SOURCE))
    parser.add_argument("--dest", default=os.getenv("COLBERT_COLLECTION", DEFAULT_DEST))
    parser.add_argument("--batch", type=int, default=16, help="Points encoded per upsert")
    parser.add_argument("--resume", action="store_true", help="Skip points that already have colbert")
    parser.add_argument("--limit", type=int, default=None, help="Encode at most N new points")
    args = parser.parse_args()

    if args.dest == args.source or args.dest == DEFAULT_SOURCE:
        raise SystemExit(
            f"Refusing to write ColBERT vectors into {args.dest!r}. "
            "Use a new collection name (default my-collection-colbert)."
        )

    client = QdrantClient(
        url=os.environ.get("CLUSTER_ENDPOINT", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=120,
        check_compatibility=False,
    )
    ensure_dest(client, args.source, args.dest)

    from fastembed import LateInteractionTextEmbedding

    print(f"loading {COLBERT_MODEL}...")
    encoder = LateInteractionTextEmbedding(COLBERT_MODEL)
    # Warm document path once.
    next(encoder.embed(["warmup document"]))

    skip = already_indexed(client, args.dest) if args.resume else set()
    if skip:
        print(f"resume: {len(skip)} points already have {COLBERT_VECTOR_NAME}")

    source_count = client.count(collection_name=args.source, exact=True).count
    print(f"source points={source_count} dest={args.dest} batch={args.batch}")

    started = time.perf_counter()
    offset = None
    seen = 0
    written = 0
    skipped = 0
    while True:
        points, offset = client.scroll(
            collection_name=args.source,
            limit=args.batch,
            offset=offset,
            with_payload=True,
            with_vectors=[DENSE_NAME, SPARSE_NAME],
        )
        if not points:
            break
        batch = []
        texts = []
        for point in points:
            seen += 1
            if point.id in skip:
                skipped += 1
                continue
            text = ((point.payload or {}).get("text") or "").strip() or " "
            texts.append(text)
            batch.append(point)
        if batch:
            matrices = list(encoder.embed(texts))
            upserts = []
            for point, matrix in zip(batch, matrices):
                vectors = point.vector or {}
                dense = vectors.get(DENSE_NAME)
                sparse = _sparse_payload(vectors.get(SPARSE_NAME))
                named = {
                    DENSE_NAME: list(dense) if dense is not None else [],
                    COLBERT_VECTOR_NAME: _as_list(matrix),
                }
                if sparse is not None:
                    named[SPARSE_NAME] = sparse
                upserts.append(
                    PointStruct(id=point.id, vector=named, payload=point.payload or {})
                )
            client.upsert(collection_name=args.dest, points=upserts)
            written += len(upserts)
            elapsed = time.perf_counter() - started
            rate = written / elapsed if elapsed else 0.0
            print(
                f"progress written={written} skipped={skipped} seen={seen}/{source_count} "
                f"{rate:.2f} pts/s rss_mib={_rss_bytes() / 1024 / 1024:.1f}"
            )
            if args.limit is not None and written >= args.limit:
                print(f"hit --limit {args.limit}")
                break
        if offset is None:
            break

    elapsed = time.perf_counter() - started
    dest_info = client.get_collection(args.dest)
    dest_count = dest_info.points_count
    print(
        f"done source={args.source} dest={args.dest} written={written} skipped={skipped} "
        f"dest_points={dest_count} elapsed_s={elapsed:.1f} "
        f"peak_rss_mib={_rss_bytes() / 1024 / 1024:.1f}"
    )
    print(
        "Qdrant Cloud does not expose per-collection disk/RAM in the public API; "
        "use the cluster dashboard for on-disk / RAM size of "
        f"{args.dest}."
    )


if __name__ == "__main__":
    main()
