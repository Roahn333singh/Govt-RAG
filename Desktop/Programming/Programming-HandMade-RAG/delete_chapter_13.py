import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("CLUSTER_ENDPOINT", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my-collection"

target_source = "/Users/rohansingh/Desktop/Programming/Programming-HandMade-RAG/Nivida docs/section_14_अध्याय_13.md"

print(f"🔌 Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

# Build a filter for the document-id (which is indexed)
from qdrant_client.models import PointIdsList

file_filter = Filter(
    must=[
        FieldCondition(
            key="document-id",
            match=MatchValue(value="Nivida Path.pdf")
        )
    ]
)

print("🔍 Scrolling through all points for 'Nivida Path.pdf'...")
matching_ids = []
offset = None

while True:
    records, offset = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=file_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False,
        offset=offset
    )
    
    for record in records:
        if record.payload and record.payload.get("source") == target_source:
            matching_ids.append(record.id)
            
    if offset is None or len(records) == 0:
        break

total_to_delete = len(matching_ids)

if total_to_delete == 0:
    print(f"⚠️ No vectors found for source: '{target_source}'")
    exit(0)

print(f"🗑️ Found {total_to_delete} vectors to delete. Proceeding...")

# Delete points by explicit IDs
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=PointIdsList(points=matching_ids)
)

print(f"✅ Successfully deleted all {total_to_delete} vectors for '{target_source}'.")
