"""Download/bake fastembed models into FASTEMBED_CACHE_PATH during image build."""

from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

print("Baking SparseTextEmbedding (BM25)...")
SparseTextEmbedding(model_name="Qdrant/bm25")
print("Baking TextCrossEncoder (MiniLM)...")
TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
print("Model bake complete.")
