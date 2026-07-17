import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition

# Load local environment variables
load_dotenv()

QDRANT_URL = os.getenv("CLUSTER_ENDPOINT", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CACHE_COLLECTION_NAME = "semantic-query-cache"

if not QDRANT_URL or not QDRANT_API_KEY:
    print("❌ Error: CLUSTER_ENDPOINT and QDRANT_API_KEY must be set in your .env file.")
    exit(1)

print(f"🔌 Connecting to Qdrant at: {QDRANT_URL}")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120.0)

# Check if the cache collection exists
if not client.collection_exists(CACHE_COLLECTION_NAME):
    print(f"⚠️  Collection '{CACHE_COLLECTION_NAME}' does not exist. Nothing to clear!")
    exit(0)

# 1. Count current cache entries
try:
    count_res = client.count(collection_name=CACHE_COLLECTION_NAME, exact=True)
    cache_count = count_res.count
    print(f"📊 Current entries in '{CACHE_COLLECTION_NAME}': {cache_count}")
except Exception as e:
    print(f"⚠️ Could not read cache count: {e}")
    cache_count = None

if cache_count == 0:
    print("🧹 Semantic cache is already empty. No deletion needed.")
    exit(0)

# 2. Ask for confirmation or proceed with deletion
print(f"🗑️ Deleting collection '{CACHE_COLLECTION_NAME}' to completely clear the cache...")
try:
    client.delete_collection(collection_name=CACHE_COLLECTION_NAME)
    print(f"✅ Successfully deleted collection '{CACHE_COLLECTION_NAME}'.")
    
    # 3. Recreate empty collection so it is immediately ready
    from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
    print(f"🛠️  Recreating empty collection '{CACHE_COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=CACHE_COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=CACHE_COLLECTION_NAME,
        field_name="language",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"✨ Successfully re-initialized '{CACHE_COLLECTION_NAME}'!")
    
except Exception as e:
    print(f"❌ Error clearing cache collection: {e}")
