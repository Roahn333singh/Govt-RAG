import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("=== CLUSTER COLLECTIONS ===")
collections = client.get_collections().collections
for col in collections:
    info = client.get_collection(col.name)
    print(f"\nCollection: '{col.name}'")
    print(f"  Points (vectors) : {info.points_count:,}")
    print(f"  Indexed vectors  : {info.indexed_vectors_count:,}")
    print(f"  Status           : {info.status}")
    print(f"  Config           : {info.config.params.vectors}")

# Calculate memory usage for existing 162 points
print("\n=== CURRENT RAM USAGE ESTIMATE ===")
print(f"  Existing 162 points @ 768d dense:")
existing_ram = 162 * 768 * 4 * 1.5 / (1024*1024)
print(f"    HNSW index: {existing_ram:.2f} MB")
print(f"    Qdrant baseline: ~100-150 MB")
print(f"    Current total: ~{existing_ram + 130:.0f} MB / 1,000 MB Free Tier limit")

print("\n=== PROJECTED AFTER FULL INGESTION ===")
target_vectors = 12000
projected_ram = (target_vectors * 768 * 4 * 1.5 / (1024*1024))
sparse_overhead = 30
baseline = 130
cache_ram = 2000 * 768 * 4 * 1.5 / (1024*1024)
total = projected_ram + sparse_overhead + baseline + cache_ram
print(f"  12,000 doc vectors (dense): {projected_ram:.1f} MB")
print(f"  2,000 cache vectors:        {cache_ram:.1f} MB")
print(f"  Sparse BM25 overhead:       {sparse_overhead} MB")
print(f"  Qdrant process baseline:    {baseline} MB")
print(f"  ─────────────────────────────────────")
print(f"  TOTAL PROJECTED:            {total:.0f} MB")
print(f"  Free Tier RAM limit:        1,000 MB")
print(f"  Headroom remaining:         {1000 - total:.0f} MB")
print(f"  Will it fit?:               {'✅ YES' if total < 1000 else '❌ NO'}")
