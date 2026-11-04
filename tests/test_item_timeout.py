"""Per-item timeout and error recording. No network."""

import asyncio

import pytest


def _item(item_id="G1", answerable=True):
    return {
        "id": item_id,
        "question": "क्या नियम है?",
        "language": "hi",
        "type": "numeric",
        "difficulty": "easy",
        "answerable": answerable,
        "source_docs": ["a.pdf"],
        "evidence_parts": ["alpha"],
    }


async def test_retrieval_error_is_a_miss_and_the_next_item_runs(monkeypatch):
    from eval import retrieval_eval
    from eval.run_eval import retrieval_summary

    calls = []

    async def fake_retrieve(question):
        calls.append(question)
        if len(calls) == 1:
            raise RuntimeError("qdrant down")
        await asyncio.sleep(0.05)
        return {
            "top_rerank_score": 1.0,
            "reranked": [],
            "merged": [],
            "rrf_by_phrasing": [],
            "kept": [],
        }

    monkeypatch.setattr(retrieval_eval, "retrieve_item", fake_retrieve)
    first, second = _item("G1"), _item("G2", answerable=False)
    rows = [
        await retrieval_eval.score_one_retrieval(1, 2, first, 60),
        await retrieval_eval.score_one_retrieval(2, 2, second, 0.01),
    ]
    assert rows[0]["status"] == "error"
    assert "RuntimeError" in rows[0]["error"]
    assert "qdrant down" in rows[0]["error"]
    assert rows[0]["recall_at_8"] == 0.0
    assert rows[0]["stages"]["reranked"]["mrr"] == 0.0
    assert rows[1]["status"] == "error"
    assert rows[1]["error"].startswith("timed out")
    assert rows[1]["recall_at_8"] is None
    summary = retrieval_summary(rows)
    assert summary["recall_at_8"] == 0.0
    assert summary["stages"]["reranked"]["mrr"] == 0.0
    assert summary["n_errors"] == 2
    assert summary["n_answerable"] == 1


async def test_keyboard_interrupt_is_not_swallowed(monkeypatch):
    """Ctrl-C is a BaseException, so the item loop must let it stop the run."""
    from eval import retrieval_eval

    class _Stop(BaseException):
        pass

    async def fake_retrieve(question):
        raise _Stop()

    monkeypatch.setattr(retrieval_eval, "retrieve_item", fake_retrieve)
    with pytest.raises(_Stop):
        await retrieval_eval.score_one_retrieval(1, 1, _item(), 60)


def test_limit_takes_the_first_n():
    from eval.run_eval import apply_limit, count_errors

    assert apply_limit(["a", "b", "c"], 2) == ["a", "b"]
    assert apply_limit(["a"], None) == ["a"]
    assert count_errors([{"status": "ok", "error": None}, {"status": "error", "error": "timed out"}]) == 1


def test_failing_items_include_the_error_message():
    from eval.upload import failing_items

    items = failing_items(
        [
            {
                "id": "G9",
                "type": "numeric",
                "language": "hi",
                "answerable": True,
                "status": "error",
                "error": "timed out after 60s",
                "question": "क्या नियम है?",
                "evidence_quote": "do not copy",
            }
        ]
    )
    assert items[0]["stage_lost"] == "error"
    assert items[0]["error"] == "timed out after 60s"
    assert items[0]["question"] == "क्या नियम है?"
    assert "evidence_quote" not in items[0]
    assert "do not copy" not in str(items[0])


def test_history_row_includes_n_errors():
    from eval.history import COLUMNS, history_row

    row = history_row({"n_errors": 2, "retrieval": {}, "history_meta": {"reranker": "minilm"}})
    assert row[COLUMNS.index("n_errors")] == "2"
    assert len(row) == len(COLUMNS)
