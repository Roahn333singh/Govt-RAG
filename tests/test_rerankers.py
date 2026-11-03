"""Reranker interface, threshold selection, and the MiniLM default. No model downloads."""

from types import SimpleNamespace

import pytest


def _docs():
    return [
        SimpleNamespace(payload={"text": "aa", "document-id": "a.pdf", "page": 1}, score=0.1, rrf_score=0.01),
        SimpleNamespace(payload={"text": "bbbb", "document-id": "b.pdf", "page": 1}, score=0.2, rrf_score=0.02),
        SimpleNamespace(payload={"text": "c", "document-id": "c.pdf", "page": 1}, score=0.3, rrf_score=0.50),
    ]


class _Encoder:
    def rerank(self, query, texts):
        del query
        return [float(len(text)) for text in texts]


class _Tokenizer:
    padding = None

    def enable_truncation(self, max_length):
        self.max_length = max_length

    def enable_padding(self, pad_id=0):
        self.pad_id = pad_id

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(ids=list(range(min(3, len(text or "")))))

    def decode(self, ids):
        return "x" * len(ids)

    def encode_batch(self, pairs):
        return [
            SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 1])
            for _ in pairs
        ]


class _Session:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, _names, feed):
        count = len(feed["input_ids"])
        return [[[float(index)] for index in range(count)]]


def _backends():
    from vectorDB.rerankers import MiniLMReranker, NoOpReranker, OnnxCrossEncoder

    return [
        MiniLMReranker(_Encoder(), "Xenova/ms-marco-MiniLM-L-6-v2"),
        OnnxCrossEncoder("mmarco", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", _Session(), _Tokenizer()),
        OnnxCrossEncoder(
            "gte",
            "Alibaba-NLP/gte-multilingual-reranker-base",
            _Session(),
            _Tokenizer(),
        ),
        NoOpReranker(),
    ]


def test_every_backend_matches_the_interface():
    docs = _docs()
    for backend in _backends():
        ranked = backend.rerank("query", docs, 2)
        assert len(ranked) <= 2
        assert ranked
        for doc, score in ranked:
            assert doc in docs
            assert isinstance(score, float)
        if backend.name == "none":
            assert [doc for doc, _score in ranked] == docs[:2]
            assert ranked[0][1] == pytest.approx(0.01)
            assert ranked[1][1] == pytest.approx(0.02)
        else:
            scores = [score for _doc, score in ranked]
            assert scores == sorted(scores, reverse=True)


def test_none_keeps_rrf_merge_order_and_fused_score():
    from vectorDB.rerankers import NoOpReranker, fused_rrf_score
    from vectorDB.search import rrf_merge

    low = SimpleNamespace(
        score=9.0,
        payload={"document-id": "low.pdf", "page": 1, "text": "low"},
    )
    high = SimpleNamespace(
        score=0.1,
        payload={"document-id": "high.pdf", "page": 1, "text": "high"},
    )
    merged = rrf_merge([[low, high], [high]], limit=2)
    assert merged[0] is high
    assert low.score == 9.0
    ranked = NoOpReranker().rerank("query", merged, 2)
    assert [doc for doc, _score in ranked] == merged
    assert ranked[0][1] == pytest.approx(fused_rrf_score(high))
    assert ranked[0][1] > ranked[1][1]


def test_default_is_minilm_with_the_current_threshold(monkeypatch):
    from vectorDB.rerankers import reject_threshold, selected_name
    from vectorDB.search import DEFAULT_RERANKER

    monkeypatch.delenv("RERANKER", raising=False)
    monkeypatch.delenv("REJECT_THRESHOLD", raising=False)
    monkeypatch.delenv("REJECT_THRESHOLD_MINILM", raising=False)
    assert selected_name() == "minilm"
    assert reject_threshold() == -10.0
    assert DEFAULT_RERANKER == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_threshold_is_selected_per_backend(monkeypatch):
    from vectorDB.rerankers import PERMISSIVE_THRESHOLD, reject_threshold

    monkeypatch.delenv("REJECT_THRESHOLD", raising=False)
    monkeypatch.setenv("RERANKER", "minilm")
    monkeypatch.setenv("REJECT_THRESHOLD_MINILM", "-3")
    assert reject_threshold() == -3.0

    monkeypatch.setenv("RERANKER", "mmarco")
    monkeypatch.delenv("REJECT_THRESHOLD_MMARCO", raising=False)
    assert reject_threshold("mmarco") == PERMISSIVE_THRESHOLD
    monkeypatch.setenv("REJECT_THRESHOLD_MMARCO", "0.4")
    assert reject_threshold("mmarco") == pytest.approx(0.4)

    monkeypatch.setenv("REJECT_THRESHOLD_GTE", "1.5")
    assert reject_threshold("gte") == pytest.approx(1.5)

    monkeypatch.delenv("COLBERT_REJECT_THRESHOLD", raising=False)
    monkeypatch.delenv("REJECT_THRESHOLD_COLBERT", raising=False)
    assert reject_threshold("colbert") == PERMISSIVE_THRESHOLD
    monkeypatch.setenv("COLBERT_REJECT_THRESHOLD", "12.5")
    assert reject_threshold("colbert") == pytest.approx(12.5)

    monkeypatch.setenv("REJECT_THRESHOLD", "0.2")
    assert reject_threshold("minilm") == pytest.approx(0.2)
    assert reject_threshold("gte") == pytest.approx(0.2)
    assert reject_threshold("colbert") == pytest.approx(0.2)
    assert reject_threshold("none") is None


def test_colbert_reranks_via_qdrant_maxsim_not_python_scores():
    from vectorDB.rerankers import COLBERT_VECTOR_NAME, ColBERTReranker

    docs = [
        SimpleNamespace(id="a", payload={"text": "aa"}, score=0.1),
        SimpleNamespace(id="b", payload={"text": "bb"}, score=0.2),
        SimpleNamespace(id="c", payload={"text": "cc"}, score=0.3),
    ]

    class _FakePoints:
        def __init__(self, points):
            self.points = points

    class _FakeClient:
        def __init__(self):
            self.kwargs = None

        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return _FakePoints(
                [
                    SimpleNamespace(id="c", score=9.0),
                    SimpleNamespace(id="a", score=4.0),
                ]
            )

    class _FakeEncoder:
        def query_embed(self, query):
            del query
            yield [[0.1, 0.2], [0.3, 0.4]]

    client = _FakeClient()
    backend = ColBERTReranker(_FakeEncoder(), "my-collection-colbert", client)
    ranked = backend.rerank("q", docs, 2)
    assert [doc.id for doc, _score in ranked] == ["c", "a"]
    assert ranked[0][1] == pytest.approx(9.0)
    assert ranked[1][1] == pytest.approx(4.0)
    assert client.kwargs["collection_name"] == "my-collection-colbert"
    assert client.kwargs["using"] == COLBERT_VECTOR_NAME
    assert client.kwargs["limit"] == 2
    assert client.kwargs["with_payload"] is False
    has_id = client.kwargs["query_filter"].must[0].has_id
    assert set(has_id) == {"a", "b", "c"}


def test_colbert_load_requires_collection(monkeypatch):
    from vectorDB.rerankers import ColBERTReranker, load_reranker

    monkeypatch.setenv("RERANKER", "colbert")
    monkeypatch.delenv("COLBERT_COLLECTION", raising=False)
    with pytest.raises(RuntimeError, match="COLBERT_COLLECTION"):
        ColBERTReranker.load()
    with pytest.raises(RuntimeError, match="Failed to load reranker 'colbert'"):
        load_reranker("colbert")


def test_gte_and_none_paths_unchanged_by_colbert_registration():
    from vectorDB.rerankers import RERANKERS, NoOpReranker, reject_threshold

    assert "colbert" in RERANKERS
    assert "gte" in RERANKERS
    assert "none" in RERANKERS
    docs = _docs()
    ranked = NoOpReranker().rerank("q", docs, 2)
    assert [doc for doc, _score in ranked] == docs[:2]
    assert ranked[0][1] == pytest.approx(0.01)


def test_unknown_and_missing_backends_do_not_fall_back(monkeypatch, tmp_path):
    from vectorDB.rerankers import load_reranker, selected_name

    monkeypatch.setenv("RERANKER", "jina")
    with pytest.raises(RuntimeError, match="fall back"):
        selected_name()

    monkeypatch.setenv("RERANKER", "mmarco")
    monkeypatch.setenv("RERANKER_CACHE_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="mmarco"):
        load_reranker()


def test_rerank_hits_writes_backend_scores_without_resorting():
    from vectorDB import search
    from vectorDB.rerankers import NoOpReranker

    docs = _docs()
    search.reranker = NoOpReranker()
    ranked = search.rerank_hits("query", docs, 2)
    assert ranked[0] is docs[0]
    assert ranked[1] is docs[1]
    assert ranked[0].score == pytest.approx(0.01)


def test_suggested_threshold_is_not_the_applied_floor():
    from vectorDB.rerankers import suggest_threshold

    records = [
        {"answerable": False, "top_score": 0.2, "relevant_scores": []},
        {"answerable": False, "top_score": 0.3, "relevant_scores": []},
        {"answerable": True, "top_score": 0.9, "relevant_scores": [0.9]},
        {"answerable": True, "top_score": 0.8, "relevant_scores": [0.8]},
    ]
    assert suggest_threshold(records) == pytest.approx(0.8)


def test_compare_table_is_side_by_side_and_does_not_apply_the_suggestion():
    from eval.rerank_compare import render_compare

    text = render_compare(
        [
            {
                "name": "minilm",
                "hit": 0.5,
                "recall": 0.5,
                "mrr": 0.4,
                "dropped": 1,
                "killed": 2,
                "applied_threshold": -10.0,
                "suggested_threshold": -2.5,
                "slices": {},
                "answerable_p50": 1.0,
                "latency_p50_ms": 12.3,
                "latency_p95_ms": 40.0,
            },
            {
                "name": "none",
                "hit": 0.25,
                "recall": 0.25,
                "mrr": 0.2,
                "dropped": 3,
                "killed": 0,
                "applied_threshold": None,
                "suggested_threshold": 0.016,
                "slices": {},
            },
        ]
    )
    assert "| minilm |" in text
    assert "| none |" in text
    assert "not applied" in text.lower() or "They are not applied." in text
    assert "skipped" in text
    header, minilm = text.splitlines()[6], text.splitlines()[8]
    assert header.count("|") == minilm.count("|")


def test_history_row_records_reranker_and_old_rows_stay_put(tmp_path):
    from eval.history import COLUMNS, append_history_row, history_row

    row = history_row({"history_meta": {"reranker": "gte"}, "retrieval": {}, "n_items": 1})
    assert row[COLUMNS.index("reranker")] == "gte"
    assert len(row) == len(COLUMNS)

    path = tmp_path / "history.csv"
    old_header = ",".join(COLUMNS[:-1])
    old_row = "old-run,2026-10-30T10:00:00+05:30,abc"
    path.write_text(old_header + "\n" + old_row + "\n", encoding="utf-8")
    append_history_row(["new-run"] + [""] * (len(COLUMNS) - 1), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert lines[1] == old_row
    assert lines[2].startswith("new-run,")
