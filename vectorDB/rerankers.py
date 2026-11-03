"""Pluggable cross-encoders. RERANKER=minilm is the production default.

Backends share rerank(query, docs, top_k) -> list[(doc, score)]. A backend that
fails to load raises. This module never substitutes a different model.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

RERANKERS = ("minilm", "mmarco", "gte", "colbert", "none")

COLBERT_MODEL = "jinaai/jina-colbert-v2"
COLBERT_DIM = 128
COLBERT_VECTOR_NAME = "colbert"

# Used only when a model-specific threshold is unset. It is not a tuned cutoff.
# The compare eval prints a suggested threshold and does not apply it.
PERMISSIVE_THRESHOLD = -1.0e9

_SPEC = {
    "mmarco": {
        "hf": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "file": "model.onnx",
    },
    "gte": {
        "hf": "Alibaba-NLP/gte-multilingual-reranker-base",
        "file": "model_quantized.onnx",
        "batch": 4,
    },
}

_RRF_SCORES: dict[int, float] = {}
_skip_logged = False


def selected_name() -> str:
    name = (os.getenv("RERANKER") or "minilm").strip().lower()
    if name not in RERANKERS:
        raise RuntimeError(
            f"Unknown RERANKER={name!r}. Expected one of {', '.join(RERANKERS)}. "
            "Refusing to fall back to another backend."
        )
    return name


def reject_threshold(name: str | None = None) -> float | None:
    """Floor for the selected backend. None means do not apply one.

    REJECT_THRESHOLD, when set, overrides the model-specific variables.
    RERANKER=none always skips the floor: RRF scores are not on a model scale.
    """
    chosen = (name or selected_name()).strip().lower()
    if chosen == "none":
        return None
    override = os.getenv("REJECT_THRESHOLD")
    if override not in (None, ""):
        return float(override)
    if chosen == "colbert":
        # MaxSim scale differs from cross-encoder logits. Prefer COLBERT_REJECT_THRESHOLD.
        for env_name in ("COLBERT_REJECT_THRESHOLD", "REJECT_THRESHOLD_COLBERT"):
            raw = os.getenv(env_name)
            if raw not in (None, ""):
                return float(raw)
        return float(PERMISSIVE_THRESHOLD)
    specific = {
        "minilm": ("REJECT_THRESHOLD_MINILM", "-10.0"),
        "mmarco": ("REJECT_THRESHOLD_MMARCO", str(PERMISSIVE_THRESHOLD)),
        "gte": ("REJECT_THRESHOLD_GTE", str(PERMISSIVE_THRESHOLD)),
    }
    if chosen not in specific:
        raise RuntimeError(
            f"Unknown RERANKER={chosen!r}. Refusing to fall back to another backend."
        )
    env_name, default = specific[chosen]
    raw = os.getenv(env_name)
    if raw in (None, ""):
        raw = default
    return float(raw)


def log_skipped_threshold() -> None:
    """One line per process when the reject floor is not applied."""
    global _skip_logged
    if _skip_logged:
        return
    _skip_logged = True
    print(
        "reject threshold skipped (RERANKER=none; RRF scores are not on a comparable scale)"
    )


def stamp_rrf(hit, score: float) -> None:
    """Remember the fused RRF score without changing hit.score."""
    value = float(score)
    try:
        setattr(hit, "rrf_score", value)
    except Exception:
        pass
    _RRF_SCORES[id(hit)] = value
    if len(_RRF_SCORES) > 8000:
        for key in list(_RRF_SCORES)[:4000]:
            _RRF_SCORES.pop(key, None)


def fused_rrf_score(hit) -> float:
    stamped = getattr(hit, "rrf_score", None)
    if stamped is not None:
        return float(stamped)
    stored = _RRF_SCORES.get(id(hit))
    if stored is not None:
        return float(stored)
    return float(getattr(hit, "score", 0) or 0)


def suggest_threshold(records: list[dict]) -> float | None:
    """Threshold that maximizes correct refusals + kept answerable items.

    Unanswerable is a correct refusal when its top score is missing or strictly
    below the candidate. Answerable is kept when any quote-matching score in the
    reranked top 8 is at or above it. Ties take the higher threshold. Callers
    print the result and do not apply it.
    """
    observed: list[float] = []
    for record in records:
        top = record.get("top_score")
        if top is not None:
            observed.append(float(top))
        observed.extend(float(score) for score in (record.get("relevant_scores") or []))
    if not observed:
        return None
    unique = sorted(set(observed))
    candidates = [unique[0] - 1.0, *unique, unique[-1] + 1.0]
    best_total = -1
    best_threshold = None
    for threshold in candidates:
        total = 0
        for record in records:
            top = record.get("top_score")
            if not record.get("answerable"):
                if top is None or float(top) < threshold:
                    total += 1
            elif any(float(score) >= threshold for score in (record.get("relevant_scores") or [])):
                total += 1
        if total > best_total or (
            total == best_total and best_threshold is not None and threshold > best_threshold
        ):
            best_total = total
            best_threshold = threshold
    return best_threshold


def cache_root() -> Path:
    base = os.getenv("RERANKER_CACHE_PATH") or os.getenv("FASTEMBED_CACHE_PATH") or "/app/models"
    return Path(base) / "rerankers"


def _max_tokens() -> int:
    from vectorDB.search import RERANK_MAX_TOKENS

    return RERANK_MAX_TOKENS


class MiniLMReranker:
    """Xenova/ms-marco-MiniLM-L-6-v2 through FastEmbed. Same path as before."""

    name = "minilm"
    batch_size = 64

    def __init__(self, encoder, model_id: str):
        self.model = encoder
        self.model_id = model_id

    @classmethod
    def load(cls) -> "MiniLMReranker":
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from vectorDB.search import DEFAULT_RERANKER, _cap_rerank_length

        model_id = os.getenv("RERANKER_MODEL") or DEFAULT_RERANKER
        encoder = TextCrossEncoder(model_name=model_id)
        _cap_rerank_length(encoder)
        return cls(encoder, model_id)

    def rerank(self, query: str, docs: list, top_k: int) -> list[tuple]:
        if not docs or top_k <= 0:
            return []
        from vectorDB.search import truncate_text, tokenizer_of

        texts = [
            truncate_text((getattr(doc, "payload", None) or {}).get("text") or "", tokenizer_of(self.model))
            for doc in docs
        ]
        scores = [float(score) for score in self.model.rerank(query, texts)]
        order = sorted(range(len(docs)), key=lambda index: scores[index], reverse=True)
        return [(docs[index], scores[index]) for index in order[:top_k]]


class OnnxCrossEncoder:
    """CPU ONNX cross-encoder. Weights come from the image, never the network."""

    def __init__(self, name: str, model_id: str, session, tokenizer, batch_size: int = 8):
        self.name = name
        self.model_id = model_id
        self.session = session
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = _max_tokens()
        if hasattr(tokenizer, "enable_truncation"):
            tokenizer.enable_truncation(max_length=self.max_length)
        pad_id = None
        padding = getattr(tokenizer, "padding", None)
        if isinstance(padding, dict) and padding.get("pad_id") is not None:
            pad_id = int(padding["pad_id"])
        if pad_id is None and hasattr(tokenizer, "token_to_id"):
            found = tokenizer.token_to_id("<pad>")
            if found is not None:
                pad_id = int(found)
        if pad_id is None:
            pad_id = 0
        if hasattr(tokenizer, "enable_padding"):
            tokenizer.enable_padding(pad_id=pad_id)

    @classmethod
    def load(cls, name: str) -> "OnnxCrossEncoder":
        spec = _SPEC[name]
        directory = cache_root() / name
        model_path = directory / spec["file"]
        tokenizer_path = directory / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"{name} files missing under {directory} "
                f"(need {spec['file']} and tokenizer.json). "
                "Bake them into the image; refusing to download at runtime."
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        providers = ort.get_available_providers()
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(f"CPUExecutionProvider is not available ({providers})")
        options = ort.SessionOptions()
        threads = int(os.getenv("ONNXRUNTIME_INTRA_OP_NUM_THREADS", "2"))
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(name, spec["hf"], session, tokenizer, batch_size=int(spec.get("batch", 8)))

    def _truncate(self, text: str) -> str:
        encoded = self.tokenizer.encode(text or "", add_special_tokens=False)
        ids = list(getattr(encoded, "ids", encoded))
        if len(ids) <= self.max_length:
            return text or ""
        return self.tokenizer.decode(ids[: self.max_length])

    def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        import numpy as np

        encoded = self.tokenizer.encode_batch([(query, text) for text in texts])
        wanted = {item.name for item in self.session.get_inputs()}
        feed = {}
        ids = [list(item.ids) for item in encoded]
        mask = [list(getattr(item, "attention_mask", []) or []) for item in encoded]
        types = [list(getattr(item, "type_ids", []) or []) for item in encoded]
        if "input_ids" in wanted:
            feed["input_ids"] = np.asarray(ids, dtype=np.int64)
        if "attention_mask" in wanted and mask and mask[0]:
            feed["attention_mask"] = np.asarray(mask, dtype=np.int64)
        width = len(ids[0]) if ids else 0
        if "token_type_ids" in wanted and width:
            if not (types and types[0]):
                types = [[0] * width for _ in texts]
            feed["token_type_ids"] = np.asarray(types, dtype=np.int64)
        if "position_ids" in wanted and width:
            feed["position_ids"] = np.tile(np.arange(width, dtype=np.int64), (len(texts), 1))
        raw = self.session.run(None, feed)[0]
        values = np.asarray(raw).reshape(len(texts), -1)
        return [float(row[0]) for row in values]

    def rerank(self, query: str, docs: list, top_k: int) -> list[tuple]:
        if not docs or top_k <= 0:
            return []
        texts = [
            self._truncate((getattr(doc, "payload", None) or {}).get("text") or "")
            for doc in docs
        ]
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            scores.extend(self._score_batch(query, texts[start : start + self.batch_size]))
        order = sorted(range(len(docs)), key=lambda index: scores[index], reverse=True)
        return [(docs[index], scores[index]) for index in order[:top_k]]


class NoOpReranker:
    """Keep rrf_merge order. The score is the fused RRF score, not re-sorted."""

    name = "none"
    model_id = "none"
    batch_size = 0

    def rerank(self, query: str, docs: list, top_k: int) -> list[tuple]:
        del query
        self.batch_size = len(docs)
        if not docs or top_k <= 0:
            return []
        return [(doc, fused_rrf_score(doc)) for doc in list(docs)[:top_k]]


class ColBERTReranker:
    """Late-interaction MaxSim via Qdrant. Scoring stays in Qdrant, not Python."""

    name = "colbert"
    model_id = COLBERT_MODEL
    batch_size = 1

    def __init__(self, encoder, collection: str, client):
        self.encoder = encoder
        self.collection = collection
        self.client = client

    @classmethod
    def load(cls) -> "ColBERTReranker":
        from fastembed import LateInteractionTextEmbedding

        from vectorDB import search

        collection = (os.getenv("COLBERT_COLLECTION") or "").strip()
        if not collection:
            raise RuntimeError(
                "COLBERT_COLLECTION is not set. Point it at the ColBERT-indexed "
                "collection (never the live my-collection). Refusing to fall back."
            )
        encoder = LateInteractionTextEmbedding(COLBERT_MODEL)
        return cls(encoder, collection, search.qdrant_client)

    def encode_query(self, query: str) -> list[list[float]]:
        matrix = next(self.encoder.query_embed(query or " "))
        if hasattr(matrix, "tolist"):
            return matrix.tolist()
        return [list(row) if not hasattr(row, "tolist") else row.tolist() for row in matrix]

    def rerank(self, query: str, docs: list, top_k: int) -> list[tuple]:
        if not docs or top_k <= 0:
            return []
        from qdrant_client.models import Filter, HasIdCondition

        ids = [doc.id for doc in docs if getattr(doc, "id", None) is not None]
        if not ids:
            raise RuntimeError(
                "ColBERT rerank needs point ids on candidates. "
                "Hybrid hits must keep their Qdrant ids."
            )
        self.batch_size = len(ids)
        query_vec = self.encode_query(query)
        ranked = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            using=COLBERT_VECTOR_NAME,
            query_filter=Filter(must=[HasIdCondition(has_id=ids)]),
            limit=min(top_k, len(ids)),
            with_payload=False,
        ).points
        by_id = {doc.id: doc for doc in docs if getattr(doc, "id", None) is not None}
        out: list[tuple] = []
        for hit in ranked:
            doc = by_id.get(hit.id)
            if doc is None:
                continue
            out.append((doc, float(hit.score or 0.0)))
        if len(out) < min(top_k, len(docs)):
            seen = {doc.id for doc, _score in out}
            for doc in docs:
                if getattr(doc, "id", None) in seen:
                    continue
                out.append((doc, float(getattr(doc, "score", 0.0) or 0.0)))
                if len(out) >= top_k:
                    break
        return out[:top_k]


def _construct(name: str):
    if name == "minilm":
        return MiniLMReranker.load()
    if name == "none":
        return NoOpReranker()
    if name == "colbert":
        return ColBERTReranker.load()
    if name in _SPEC:
        return OnnxCrossEncoder.load(name)
    raise RuntimeError(
        f"Unknown RERANKER={name!r}. Refusing to fall back to another backend."
    )


def _warmup(backend) -> None:
    if getattr(backend, "name", None) == "colbert":
        # Query encoder only. MaxSim needs real point ids in COLBERT_COLLECTION.
        backend.encode_query("warmup")
        return
    dummy = SimpleNamespace(payload={"text": "warmup"}, score=0.0, rrf_score=0.0)
    backend.rerank("warmup", [dummy], 1)


def load_reranker(name: str | None = None):
    """Load the requested backend once. Failure raises; there is no fallback."""
    chosen = (name or selected_name()).strip().lower()
    try:
        backend = _construct(chosen)
        _warmup(backend)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load reranker {chosen!r}. Refusing to fall back to another backend."
        ) from exc
    return backend
