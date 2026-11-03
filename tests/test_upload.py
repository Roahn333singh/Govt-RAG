"""GCS upload payload builders. No network and no bucket."""

import json
from pathlib import Path

from google.api_core.exceptions import NotFound


def _csv(cells: dict) -> str:
    from eval.history import COLUMNS

    header = ",".join(COLUMNS)
    row = []
    for column in COLUMNS:
        value = cells.get(column, "")
        text = "" if value is None else str(value)
        if any(ch in text for ch in ",\"\n"):
            text = '"' + text.replace('"', '""') + '"'
        row.append(text)
    return header + "\n" + ",".join(row) + "\n"


def test_history_json_keeps_numbers_nulls_and_hindi():
    from eval.upload import history_records

    text = _csv(
        {
            "run_id": "run-1",
            "timestamp_ist": "2026-10-30T16:18:34+05:30",
            "n_items": "25",
            "n_answerable": "21",
            "n_refusal": "",
            "recall_at_8": "0.5000",
            "hit_at_8": "0.0000",
            "latency_p50_ms": "1200",
            "latency_p95_ms": "",
            "cost_usd": "",
            "faithfulness_avg": "",
            "gate_failures": "हिंदी में असफल",
            "reranker": "minilm",
        }
    )
    records = history_records(text)
    assert len(records) == 1
    row = records[0]
    assert row["n_items"] == 25
    assert isinstance(row["n_items"], int)
    assert row["recall_at_8"] == 0.5
    assert isinstance(row["recall_at_8"], float)
    assert row["hit_at_8"] == 0.0
    assert row["latency_p50_ms"] == 1200
    assert row["n_refusal"] is None
    assert row["latency_p95_ms"] is None
    assert row["cost_usd"] is None
    assert row["faithfulness_avg"] is None
    assert row["gate_failures"] == "हिंदी में असफल"
    encoded = json.dumps(records, ensure_ascii=False)
    assert "हिंदी में असफल" in encoded
    assert "\\u0939" not in encoded


def test_failing_items_use_the_stage_that_lost_the_quote():
    from eval.upload import failing_items

    def stages(rrf=1, pool=1, top=1, kept=1):
        return {
            "rrf_candidates": {"recall": rrf},
            "merged": {"recall": pool},
            "reranked": {"recall": top},
            "kept": {"recall": kept},
        }

    rows = [
        {"id": "keep", "answerable": True, "type": "numeric", "language": "en", "stages": stages(), "top_rerank_score": 1.2},
        {"id": "at-rrf", "answerable": True, "type": "numeric", "language": "hi", "stages": stages(rrf=0), "top_rerank_score": None, "evidence_quote": "secret quote", "answer": "x" * 400},
        {"id": "at-pool", "answerable": True, "type": "single_fact", "language": "en", "stages": stages(pool=0), "top_rerank_score": 0.2},
        {"id": "at-rerank", "answerable": True, "type": "paraphrase_dense", "language": "en", "stages": stages(top=0.5), "top_rerank_score": -1.0},
        {"id": "at-threshold", "answerable": True, "type": "procedural", "language": "en", "stages": stages(kept=0), "top_rerank_score": -3.0},
        {
            "id": "refused",
            "answerable": True,
            "type": "comparison",
            "language": "hi",
            "stages": stages(),
            "top_rerank_score": 2.0,
            "judge": {"refusal": True},
            "answer": "Bearer ya29.super-secret-token and the rest",
            "question": "क्या नियम है?",
        },
        {"id": "unanswerable", "answerable": False, "type": "unanswerable", "language": "en", "judge": {"refusal": True}, "stages": {}},
    ]
    items = failing_items(rows, {"at-rrf": "प्रश्न"})
    by_id = {item["id"]: item for item in items}
    assert "keep" not in by_id
    assert "unanswerable" not in by_id
    assert [item["stage_lost"] for item in items] == ["rrf", "pool20", "rerank", "threshold", "refused"]
    assert by_id["at-rrf"]["top_score"] is None
    assert by_id["at-rrf"]["question"] == "प्रश्न"
    assert by_id["at-rrf"]["answer_excerpt"] is not None
    assert len(by_id["at-rrf"]["answer_excerpt"]) == 300
    assert "evidence_quote" not in by_id["at-rrf"]
    assert "secret quote" not in json.dumps(by_id["at-rrf"])
    assert "ya29" not in by_id["refused"]["answer_excerpt"]
    assert "[redacted]" in by_id["refused"]["answer_excerpt"]
    assert by_id["refused"]["question"] == "क्या नियम है?"
    assert set(by_id["at-pool"]) == {
        "id",
        "type",
        "language",
        "question",
        "stage_lost",
        "top_score",
        "answer_excerpt",
    }


def test_latest_json_is_uploaded_last_and_a_missing_bucket_is_skipped(tmp_path: Path):
    from eval.upload import upload_runs

    class _Blob:
        def __init__(self, name: str, log: list[str], fail: bool):
            self.name = name
            self.log = log
            self.fail = fail

        def upload_from_string(self, data, content_type=None):
            if self.fail and self.name.endswith("summary.json"):
                raise NotFound("The specified bucket does not exist.")
            self.log.append(self.name)

    class _Bucket:
        def __init__(self, log: list[str], fail: bool = False):
            self.log = log
            self.fail = fail

        def blob(self, name: str):
            return _Blob(name, self.log, self.fail)

    class _Client:
        def __init__(self, log: list[str], fail: bool = False):
            self.log = log
            self.fail = fail

        def bucket(self, name: str):
            assert name == "govt-rag-evals"
            return _Bucket(self.log, self.fail)

    history = tmp_path / "history.csv"
    history.write_text(_csv({"run_id": "old", "n_items": "1", "gate_failures": "हिंदी"}), encoding="utf-8")
    runs = [
        {
            "run_id": "first",
            "timestamp_ist": "2026-10-30T16:00:00+05:30",
            "document": {"run_id": "first"},
            "report_md": "# one\n",
            "compare_md": None,
            "failing": [],
        },
        {
            "run_id": "second",
            "timestamp_ist": "2026-10-30T17:00:00+05:30",
            "document": {"run_id": "second"},
            "report_md": None,
            "compare_md": "# compare\n",
            "failing": [],
        },
    ]
    log: list[str] = []
    assert upload_runs(runs, history_path=history, client=_Client(log)) is True
    assert log[-1] == "latest.json"
    assert log.index("history.json") < log.index("latest.json")
    assert log.index("runs/second/compare.md") < log.index("latest.json")
    assert "runs/first/summary.json" in log
    assert "runs/second/summary.json" in log
    assert "history.csv" in log

    missed: list[str] = []
    assert upload_runs(runs[:1], history_path=history, client=_Client(missed, fail=True)) is False
    assert "latest.json" not in missed


def test_missing_metrics_stay_null():
    from eval.upload import monitor_summary

    document = monitor_summary(
        {
            "n_items": 4,
            "retrieval_set": "sample25",
            "answer_set": None,
            "retrieval": {"recall_at_8": 0.5, "stages": {"reranked": {"hit": 0.5, "mrr": 0.25, "recall": 0.5}, "merged": {"recall": 0.75}}},
            "answer": None,
            "slices": {"type": {"numeric": {"recall_at_8": 1.0}, "unanswerable": {"recall_at_8": None}}, "language": {"en": {"recall_at_8": 0.5}, "hi": {}}},
            "deltas": {"recall_at_8": None},
            "gate_result": [],
            "history_meta": {
                "run_id": "abc",
                "timestamp_ist": "2026-10-30T16:18:34+05:30",
                "git_branch": "main",
                "reranker": "minilm",
                "cloud_run_revision": "",
            },
            "git_sha": "732b0b7",
        }
    )
    assert document["metrics"]["faithfulness_avg"] is None
    assert document["metrics"]["cost_usd"] is None
    assert document["metrics"]["latency_p95_ms"] is None
    assert document["metrics"]["recall_at_8_hi"] is None
    assert document["metrics"]["recall_at_8"] == 0.5
    assert document["deltas"] == {"recall_at_8": None}
    assert document["gate"] == {"passed": True, "failures": []}
    assert document["run"]["cloud_run_revision"] is None
    assert document["run"]["judge_model"] is None
    assert [stage["stage"] for stage in document["stages"]] == [
        "RRF candidates",
        "Pool 20",
        "Top 8",
        "After threshold",
    ]
    assert document["types"]["unanswerable"] is None
    assert document["types"]["numeric"] == 1.0
