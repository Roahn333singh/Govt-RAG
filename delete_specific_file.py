import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("CLUSTER_ENDPOINT", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my-collection"

if len(sys.argv) < 2:
    print("Usage: python delete_specific_file.py <filename>")
    sys.exit(1)

target_file = sys.argv[1]

print(f"🔌 Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

# Build a filter for the specific file
file_filter = Filter(
    must=[
        FieldCondition(
            key="document-id",
            match=MatchValue(value=target_file)
        )
    ]
)

# First, let's count how many we are deleting
count_result = client.count(
    collection_name=COLLECTION_NAME,
    count_filter=file_filter
)
total_to_delete = count_result.count

if total_to_delete == 0:
    print(f"⚠️  No vectors found for document-id: '{target_file}'")
    sys.exit(0)

print(f"🗑️  Found {total_to_delete} vectors for '{target_file}'. Deleting...")

# Delete points matching the filter
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=file_filter
)

print(f"✅ Successfully deleted all {total_to_delete} vectors for '{target_file}'.")
