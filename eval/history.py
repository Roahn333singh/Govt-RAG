"""Append-only run log. Reads a finished summary; does not recompute metrics."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "eval" / "history.csv"
RESULTS = ROOT / "eval" / "results"
GOLDEN = ROOT / "eval" / "golden200.csv"
IST = ZoneInfo("Asia/Kolkata")

# golden200.csv `type` -> history column. Types with no column are omitted.
TYPE_COLUMNS = {
    "numeric": "recall_at_8_numeric",
    "single_fact": "recall_at_8_single_fact",
    "paraphrase_dense": "recall_at_8_paraphrase",
    "exact_term_bm25": "recall_at_8_exact_term",
    "multi_hop": "recall_at_8_multi_hop",
    "procedural": "recall_at_8_procedural",
    "comparison": "recall_at_8_comparison",
}

COLUMNS = [
    "run_id",
    "timestamp_ist",
    "git_sha",
    "git_branch",
    "pr_number",
    "cloud_run_revision",
    "target_url_kind",
    "eval_set",
    "dataset_version",
    "judge_model",
    "judge_prompt_version",
    "n_items",
    "n_answerable",
    "n_refusal",
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
    "latency_p50_ms",
    "latency_p95_ms",
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
    "gate_passed",
    "gate_failures",
    "results_dir",
    "reranker",
    "n_errors",
    "variant",
]


def dataset_version(path: Path | None = None) -> str:
    raw = (path or GOLDEN).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:12]


def url_kind(url: str | None) -> str:
    forced = (os.getenv("EVAL_URL_KIND") or "").strip().lower()
    if forced in {"local", "tagged", "prod"}:
        return forced
    text = (url or "").lower()
    if not text or "localhost" in text or "127.0.0.1" in text:
        return "local"
    if "---" in text:
        return "tagged"
    return "prod"


def _blank(value) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def _rate(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def _ms(seconds) -> str:
    if seconds is None or seconds == "":
        return ""
    return str(int(round(float(seconds) * 1000)))


def _cost(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def _div(numerator, denominator) -> str:
    if numerator is None or not denominator:
        return ""
    return _rate(float(numerator) / float(denominator))


def _slice(summary: dict, key: str, name: str, field: str):
    group = ((summary.get("slices") or {}).get(key) or {}).get(name) or {}
    return group.get(field)


def history_row(summary: dict) -> list[str]:
    """One CSV row from an already-built summary. Missing measures stay blank."""
    meta = summary.get("history_meta") or {}
    retrieval = summary.get("retrieval") or {}
    answer = summary.get("answer") or {}
    stages = retrieval.get("stages") or {}
    reranked = stages.get("reranked") or {}
    merged = stages.get("merged") or {}
    n_answerable = retrieval.get("n_answerable")
    if n_answerable is None:
        n_answerable = meta.get("n_answerable")
    failures = summary.get("gate_result")
    if failures is None:
        failures = summary.get("gate") or []
    failure_text = " | ".join(str(item).replace(";", ",") for item in failures)
    retrieval_set = summary.get("retrieval_set")
    answer_set = summary.get("answer_set")
    if retrieval_set and answer_set and retrieval_set != answer_set:
        which = retrieval_set
    else:
        which = retrieval_set or answer_set or ""
    cells = {
        "run_id": _blank(meta.get("run_id")),
        "timestamp_ist": _blank(meta.get("timestamp_ist")),
        "git_sha": _blank(summary.get("git_sha") or meta.get("git_sha")),
        "git_branch": _blank(meta.get("git_branch")),
        "pr_number": _blank(meta.get("pr_number")),
        "cloud_run_revision": _blank(meta.get("cloud_run_revision")),
        "target_url_kind": _blank(meta.get("target_url_kind")),
        "eval_set": _blank(which),
        "dataset_version": _blank(meta.get("dataset_version")),
        "judge_model": _blank((answer or {}).get("judge_model") or meta.get("judge_model")),
        "judge_prompt_version": _blank(meta.get("judge_prompt_version")),
        "n_items": _blank(summary.get("n_items") if summary.get("n_items") is not None else meta.get("n_items")),
        "n_answerable": _blank(n_answerable),
        "n_refusal": _blank(meta.get("n_refusal")),
        "recall_at_8": _rate(retrieval.get("recall_at_8")),
        "hit_at_8": _rate(reranked.get("hit")),
        "mrr": _rate(reranked.get("mrr")),
        "recall_pool20": _rate(merged.get("recall")),
        "reranker_drop_rate": _div(retrieval.get("dropped_by_reranker"), n_answerable),
        "threshold_kill_rate": _div(retrieval.get("killed_by_threshold"), n_answerable),
        "correctness_avg": _rate(answer.get("correctness_mean") if answer else None),
        "completeness_avg": _rate(answer.get("completeness_mean") if answer else None),
        "faithfulness_avg": _rate(answer.get("faithfulness") if answer else None),
        "language_match_rate": _rate(answer.get("language_match_rate") if answer else None),
        "correct_refusal_rate": _rate(answer.get("correct_refusal_rate") if answer else None),
        "false_refusal_rate": _rate(answer.get("false_refusal_rate") if answer else None),
        "injection_resist_rate": _rate(
            answer.get("adversarial_injection_pass_rate") if answer else None
        ),
        "latency_p50_ms": _ms(answer.get("latency_p50_s") if answer else None),
        "latency_p95_ms": _ms(answer.get("latency_p95_s") if answer else None),
        "error_rate": _rate(answer.get("error_rate") if answer else None),
        "cost_usd": _cost(answer.get("cost_usd") if answer else None),
        "recall_at_8_en": _rate(_slice(summary, "language", "en", "recall_at_8")),
        "recall_at_8_hi": _rate(_slice(summary, "language", "hi", "recall_at_8")),
        "faithfulness_en": _rate(_slice(summary, "language", "en", "faithfulness")),
        "faithfulness_hi": _rate(_slice(summary, "language", "hi", "faithfulness")),
        "gate_passed": "true" if not failures else "false",
        "gate_failures": failure_text,
        "results_dir": _blank(meta.get("results_dir")),
        "reranker": _blank(meta.get("reranker") or summary.get("reranker") or "minilm"),
        "n_errors": _blank(summary.get("n_errors")),
        "variant": _blank(meta.get("variant") or summary.get("variant")),
    }
    for type_name, column in TYPE_COLUMNS.items():
        cells[column] = _rate(_slice(summary, "type", type_name, "recall_at_8"))
    return [cells[column] for column in COLUMNS]


def _write_rows(path: Path, previous: str, new_lines: list[list[str]]) -> None:
    """Copy existing text unchanged, then add rows, via a temp file in the same directory."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if not previous.strip():
        writer.writerow(COLUMNS)
    for line in new_lines:
        writer.writerow(line)
    payload = previous
    if payload and not payload.endswith("\n"):
        payload += "\n"
    if not previous.strip():
        payload = ""
    payload += buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _with_reranker_column(previous: str) -> str:
    """Append header columns added after the file was created.

    Data rows are left as they are. New rows carry the extra cells.
    """
    if not previous.strip():
        return previous
    ends_with_newline = previous.endswith("\n")
    body = previous[:-1] if ends_with_newline else previous
    first, sep, rest = body.partition("\n")
    fields = next(csv.reader([first]))
    missing = [column for column in COLUMNS if column not in fields]
    if not missing:
        return previous
    updated = first + "".join("," + column for column in missing)
    if sep:
        updated += "\n" + rest
    if ends_with_newline:
        updated += "\n"
    return updated


def append_history_row(row: list[str], path: Path | None = None) -> None:
    """Add one row. Existing rows are copied through unchanged."""
    path = path or HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(".history.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            previous = path.read_text(encoding="utf-8") if path.exists() else ""
            _write_rows(path, _with_reranker_column(previous), [row])
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ist_now() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def ist_from_summary_timestamp(stamp: str | None) -> str:
    if not stamp:
        return ""
    try:
        parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return stamp
    return parsed.astimezone(IST).isoformat(timespec="seconds")


def backfill_history(results_dir: Path | None = None, path: Path | None = None) -> int:
    """Replace history.csv with one row per eval/results/*/summary.json, oldest first."""
    results_dir = results_dir or RESULTS
    path = path or HISTORY
    folders = []
    if results_dir.exists():
        folders = sorted(
            folder
            for folder in results_dir.iterdir()
            if folder.is_dir() and (folder / "summary.json").is_file()
        )
    rows = []
    for folder in folders:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        meta = dict(summary.get("history_meta") or {})
        meta.setdefault("run_id", folder.name)
        meta.setdefault("timestamp_ist", ist_from_summary_timestamp(summary.get("timestamp")))
        meta.setdefault("results_dir", str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder))
        summary["history_meta"] = meta
        rows.append(history_row(summary))
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(".history.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _write_rows(path, "", rows)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return len(rows)
