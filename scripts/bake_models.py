"""Download/bake fastembed models into FASTEMBED_CACHE_PATH during image build."""

from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

print("Baking SparseTextEmbedding (BM25)...")
SparseTextEmbedding(model_name="Qdrant/bm25")
print("Baking TextCrossEncoder (jina reranker v2 multilingual)...")
TextCrossEncoder(model_name="jinaai/jina-reranker-v2-base-multilingual")
print("Model bake complete.")
