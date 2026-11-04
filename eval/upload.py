"""Upload a finished eval run to gs://govt-rag-evals.

Reads the summary that run_eval already computed. Does not recompute metrics.
A missing bucket or any upload error is a warning. The eval result is unchanged.
This module never creates the bucket.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

BUCKET = "govt-rag-evals"
PROJECT = "project-0ff01dde-a087-4c96-b04"

_INT_FIELDS = {
    "n_items",
    "n_answerable",
    "n_refusal",
    "n_errors",
    "latency_p50_ms",
    "latency_p95_ms",
}
_FLOAT_FIELDS = {
    "recall_at_8",
    "hit_at_8",
    "mrr",
    "recall_pool20",
    "reranker_drop_rate",
    "threshold_kill_rate",
    "correctness_avg",
    "completeness_avg",
    "faithfulness_avg",
    "language_match_rate",
    "correct_refusal_rate",
    "false_refusal_rate",
    "injection_resist_rate",
    "error_rate",
    "cost_usd",
    "recall_at_8_en",
    "recall_at_8_hi",
    "faithfulness_en",
    "faithfulness_hi",
    "recall_at_8_numeric",
    "recall_at_8_single_fact",
    "recall_at_8_paraphrase",
    "recall_at_8_exact_term",
    "recall_at_8_multi_hop",
    "recall_at_8_procedural",
    "recall_at_8_comparison",
}
_STAGE_LABELS = (
    ("rrf_candidates", "RRF candidates"),
    ("merged", "Pool 20"),
    ("reranked", "Top 8"),
    ("kept", "After threshold"),
)
_LOST_AT = (
    ("rrf_candidates", "rrf"),
    ("merged", "pool20"),
    ("reranked", "rerank"),
    ("kept", "threshold"),
)
_SECRET = re.compile(
    r"(?i)(bearer\s+[A-Za-z0-9._\-]{12,}|sk-[A-Za-z0-9]{8,}|ya29\.[A-Za-z0-9._\-]+|AIza[0-9A-Za-z\-_]{20,}|xox[baprs]-[A-Za-z0-9\-]+)"
)
_JSON = "application/json; charset=utf-8"


def should_upload(no_upload: bool = False) -> bool:
    """Pull requests never upload. --no-upload skips a local or CI run."""
    if no_upload:
        return False
    if os.getenv("GITHUB_EVENT_NAME") == "pull_request":
        print("warning: not uploading eval results from a pull request")
        return False
    return True


def _null_number(value, kind: str):
    if value is None or value == "":
        return None
    if kind == "int":
        return int(value)
    return float(value)


def history_records(csv_text: str) -> list[dict]:
    """One object per history.csv row. Numbers stay numbers. Blanks are null."""
    rows = []
    for record in csv.DictReader(csv_text.splitlines()):
        parsed = {}
        for key, raw in record.items():
            if key is None:
                continue
            text = "" if raw is None else str(raw)
            if key in _INT_FIELDS:
                parsed[key] = _null_number(text, "int")
            elif key in _FLOAT_FIELDS:
                parsed[key] = _null_number(text, "float")
            else:
                parsed[key] = text if text != "" else None
        rows.append(parsed)
    return rows


def _clean(text: str | None, limit: int | None = None) -> str | None:
    if text is None:
        return None
    cleaned = _SECRET.sub("[redacted]", str(text)).strip()
    if not cleaned:
        return None
    if limit is not None:
        cleaned = cleaned[:limit]
    return cleaned


def _stage_lost(row: dict) -> str | None:
    """Earliest stage that lost recall, or a false refusal after a full retrieval."""
    stages = row.get("stages") or {}
    previous = 1.0
    for key, label in _LOST_AT:
        recall = (stages.get(key) or {}).get("recall")
        if recall is None:
            continue
        if float(recall) < previous:
            return label
        previous = float(recall)
    judge = row.get("judge") or {}
    if row.get("answerable") and judge.get("refusal") is True:
        return "refused"
    return None


def _item_error(row: dict) -> str | None:
    answer_error = row.get("answer_error")
    retrieval_error = row.get("error")
    retrieval_text = retrieval_error if isinstance(retrieval_error, str) and retrieval_error else ""
    if answer_error and retrieval_text:
        return f"{retrieval_text}; {answer_error}"
    if answer_error:
        return str(answer_error)
    if row.get("status") == "error":
        return str(retrieval_error or "error")
    if retrieval_text:
        return retrieval_text
    return None


def _failing_record(row: dict, questions: dict[str, str], stage_lost: str, error: str | None) -> dict:
    score = row.get("top_rerank_score")
    question = row.get("question") if row.get("question") else questions.get(row.get("id") or "")
    record = {
        "id": row.get("id"),
        "type": row.get("type"),
        "language": row.get("language"),
        "question": _clean(question),
        "stage_lost": stage_lost,
        "top_score": None if score is None else float(score),
        "answer_excerpt": _clean(row.get("answer"), limit=300),
    }
    if error:
        record["error"] = _clean(error, limit=300)
    return record


def failing_items(rows: list[dict], questions: dict[str, str] | None = None) -> list[dict]:
    """Items that missed the top 8, died at the threshold, were wrongly refused, or errored.

    The result has only the monitor fields. evidence_quote is never copied.
    """
    questions = questions or {}
    found = []
    for row in rows:
        message = _item_error(row)
        if message:
            found.append(_failing_record(row, questions, "error", message))
            continue
        if not row.get("answerable"):
            continue
        lost = _stage_lost(row)
        if lost is None:
            continue
        found.append(_failing_record(row, questions, lost, None))
    return found


def _eval_set(summary: dict) -> str | None:
    retrieval_set = summary.get("retrieval_set")
    answer_set = summary.get("answer_set")
    if retrieval_set and answer_set and retrieval_set != answer_set:
        which = retrieval_set
    else:
        which = retrieval_set or answer_set or ""
    return which or None


def _metric(summary: dict, name: str):
    retrieval = summary.get("retrieval") or {}
    answer = summary.get("answer") or {}
    stages = retrieval.get("stages") or {}
    reranked = stages.get("reranked") or {}
    merged = stages.get("merged") or {}
    slices = summary.get("slices") or {}
    language = slices.get("language") or {}

    def lang(code: str):
        return _null_number(((language.get(code) or {}).get("recall_at_8")), "float")

    values = {
        "recall_at_8": _null_number(retrieval.get("recall_at_8"), "float"),
        "hit_at_8": _null_number(reranked.get("hit"), "float"),
        "mrr": _null_number(reranked.get("mrr"), "float"),
        "recall_pool20": _null_number(merged.get("recall"), "float"),
        "faithfulness_avg": _null_number(answer.get("faithfulness") if answer else None, "float"),
        "correctness_avg": _null_number(answer.get("correctness_mean") if answer else None, "float"),
        "completeness_avg": _null_number(answer.get("completeness_mean") if answer else None, "float"),
        "language_match_rate": _null_number(answer.get("language_match_rate") if answer else None, "float"),
        "correct_refusal_rate": _null_number(answer.get("correct_refusal_rate") if answer else None, "float"),
        "false_refusal_rate": _null_number(answer.get("false_refusal_rate") if answer else None, "float"),
        "latency_p50_ms": _ms(answer.get("latency_p50_s") if answer else None),
        "latency_p95_ms": _ms(answer.get("latency_p95_s") if answer else None),
        "error_rate": _null_number(answer.get("error_rate") if answer else None, "float"),
        "cost_usd": _null_number(answer.get("cost_usd") if answer else None, "float"),
        "recall_at_8_en": lang("en"),
        "recall_at_8_hi": lang("hi"),
    }
    return values[name]


def _ms(seconds):
    if seconds is None or seconds == "":
        return None
    return int(round(float(seconds) * 1000))


def monitor_summary(summary: dict, thresholds: dict | None = None) -> dict:
    """Existing summary plus the keys the monitor reads. Metrics are copied, not recomputed."""
    from eval.gate import load_thresholds

    meta = summary.get("history_meta") or {}
    document = dict(summary)
    failures = summary.get("gate_result")
    if failures is None:
        failures = summary.get("gate") or []
    failures = [str(item) for item in failures]
    raw_delta = (summary.get("deltas") or {}).get("recall_at_8")
    judge = (summary.get("answer") or {}).get("judge_model") or meta.get("judge_model") or None
    if judge == "":
        judge = None
    revision = meta.get("cloud_run_revision") or None
    if revision == "":
        revision = None
    document["run"] = {
        "run_id": meta.get("run_id"),
        "timestamp_ist": meta.get("timestamp_ist"),
        "git_sha": summary.get("git_sha") or meta.get("git_sha"),
        "git_branch": meta.get("git_branch") or None,
        "reranker": meta.get("reranker") or summary.get("reranker") or "minilm",
        "variant": meta.get("variant") or summary.get("variant") or None,
        "eval_set": _eval_set(summary),
        "n_items": summary.get("n_items"),
        "judge_model": judge,
        "cloud_run_revision": revision,
    }
    document["metrics"] = {name: _metric(summary, name) for name in (
        "recall_at_8",
        "hit_at_8",
        "mrr",
        "recall_pool20",
        "faithfulness_avg",
        "correctness_avg",
        "completeness_avg",
        "language_match_rate",
        "correct_refusal_rate",
        "false_refusal_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "error_rate",
        "cost_usd",
        "recall_at_8_en",
        "recall_at_8_hi",
    )}
    stages = (summary.get("retrieval") or {}).get("stages") or {}
    document["stages"] = [
        {"stage": label, "recall": _null_number((stages.get(key) or {}).get("recall"), "float")}
        for key, label in _STAGE_LABELS
    ]
    document["types"] = {
        name: _null_number((metrics or {}).get("recall_at_8"), "float")
        for name, metrics in ((summary.get("slices") or {}).get("type") or {}).items()
        if name not in ("", "None")
    }
    document["thresholds"] = thresholds if thresholds is not None else load_thresholds()
    document["gate"] = {"passed": not failures, "failures": failures}
    document["deltas"] = {
        "recall_at_8": None if raw_delta is None else round(float(raw_delta) * 100.0, 4)
    }
    return document


def _dumps(payload) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _put(bucket, name: str, data: bytes, content_type: str) -> None:
    blob = bucket.blob(name)
    blob.upload_from_string(data, content_type=content_type)


def _questions() -> dict[str, str]:
    from eval.scoring import load_items

    return {item["id"]: item["question"] for item in load_items("full")}


def prepare_run(summary: dict, rows: list[dict], results_dir: Path, questions: dict[str, str]) -> dict:
    meta = summary.get("history_meta") or {}
    report_path = results_dir / "report.md"
    compare_path = results_dir / "compare.md"
    return {
        "run_id": meta.get("run_id") or "",
        "timestamp_ist": meta.get("timestamp_ist") or "",
        "document": monitor_summary(summary),
        "report_md": report_path.read_text(encoding="utf-8") if report_path.is_file() else None,
        "compare_md": compare_path.read_text(encoding="utf-8") if compare_path.is_file() else None,
        "failing": failing_items(rows, questions),
    }


def upload_runs(runs: list[dict], *, history_path: Path, client=None) -> bool:
    """Upload run objects, then history, then latest.json. Never raises."""
    if not runs:
        return False
    try:
        from google.api_core.exceptions import NotFound
    except ImportError:
        print("warning: google-cloud-storage is not installed; skipping eval upload")
        return False
    try:
        storage_client = client
        if storage_client is None:
            from google.cloud import storage

            storage_client = storage.Client(project=PROJECT)
        bucket = storage_client.bucket(BUCKET)
        for run in runs:
            prefix = f"runs/{run['run_id']}"
            _put(bucket, f"{prefix}/summary.json", _dumps(run["document"]), _JSON)
            if run.get("report_md") is not None:
                _put(bucket, f"{prefix}/report.md", run["report_md"].encode("utf-8"), "text/markdown; charset=utf-8")
            if run.get("compare_md") is not None:
                _put(
                    bucket,
                    f"{prefix}/compare.md",
                    run["compare_md"].encode("utf-8"),
                    "text/markdown; charset=utf-8",
                )
            _put(bucket, f"{prefix}/failing_items.json", _dumps(run["failing"]), _JSON)
        csv_bytes = history_path.read_bytes()
        _put(bucket, "history.csv", csv_bytes, "text/csv; charset=utf-8")
        records = history_records(csv_bytes.decode("utf-8"))
        _put(bucket, "history.json", _dumps(records), _JSON)
        last = runs[-1]
        _put(
            bucket,
            "latest.json",
            _dumps(
                {
                    "run_id": last["run_id"],
                    "timestamp_ist": last["timestamp_ist"],
                    "path": f"runs/{last['run_id']}/",
                }
            ),
            _JSON,
        )
        print(f"uploaded gs://{BUCKET}/runs/{last['run_id']}/")
        return True
    except NotFound:
        print(f"warning: gs://{BUCKET} is missing; skipping eval upload")
        return False
    except Exception as exc:
        print(f"warning: eval upload failed ({type(exc).__name__})")
        return False


def publish(pairs: list[tuple[dict, list]], results_dir: Path, *, no_upload: bool = False) -> bool:
    """Upload one or more finished runs from this process. Safe to call at the end of run_eval."""
    if not should_upload(no_upload):
        return False
    from eval.history import HISTORY

    try:
        questions = _questions()
        runs = [prepare_run(summary, rows, results_dir, questions) for summary, rows in pairs]
        return upload_runs(runs, history_path=HISTORY)
    except Exception as exc:
        print(f"warning: eval upload failed ({type(exc).__name__})")
        return False
