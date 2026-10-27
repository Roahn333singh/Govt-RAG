#!/usr/bin/env python3
"""Purge poisoned / injection-tainted entries from semantic-query-cache.

Usage:
  .venv/bin/python scripts/purge_cache.py --dry-run          # list known probes
  .venv/bin/python scripts/purge_cache.py                    # delete known probes
  .venv/bin/python scripts/purge_cache.py --contains "50 GB" # delete by substring
  .venv/bin/python scripts/purge_cache.py --wipe             # delete entire cache
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(override=True)

from qdrant_client import QdrantClient

from vectorDB.cache_admin import POISON_MARKERS, purge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wipe", action="store_true", help="Delete the whole cache")
    parser.add_argument("--contains", action="append", help="Substring to match (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="List matches, delete nothing")
    args = parser.parse_args()

    client = QdrantClient(
        url=os.getenv("CLUSTER_ENDPOINT", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
        check_compatibility=False,
    )

    result = purge(
        client,
        needles=args.contains or POISON_MARKERS,
        wipe=args.wipe,
        dry_run=args.dry_run,
    )
    if args.wipe:
        verb = "Would wipe" if args.dry_run else "Wiped"
        print(f"{verb} {result['matched']} entries")
        return
    for m in result["matched"]:
        print(f"  {'match' if args.dry_run else 'delete'} id={m['id']} q={m['question'][:80]!r}")
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {len(result['matched'])} entries")


if __name__ == "__main__":
    main()
