"""Unit tests for the eval grader, gate, and judge payload. No network."""

import json

from eval.answer_eval import cache_key, estimate_cost, judge_user_payload
from eval.gate import gate_failures
from eval.scoring import (
    load_items,
    normalize_text,
    part_found,
    split_parts,
    stage_metrics,
)


def test_split_parts_on_the_joiner_only():
    assert split_parts("14 days in a calendar year. || 10 days in continuity") == [
        "14 days in a calendar year.",
        "10 days in continuity",
    ]
    assert split_parts("Form 27 A") == ["Form 27 A"]
    assert split_parts("") == []
    assert split_parts(None) == []
    assert split_parts("keep a single | pipe") == ["keep a single | pipe"]


def test_normalizer_strips_private_use_and_controls():
    assert normalize_text("Hello\uf097 World") == "hello world"
    assert normalize_text("Line\u0007one\nline\ttwo") == "lineone line two"
    assert normalize_text("  NFKC　space  ") == "nfkc space"


def test_quote_must_sit_in_a_matching_source():
    chunks = [
        {"source": "Other.pdf", "text": "14 days in a calendar year."},
        {"source": "Leave Rule PPT.pdf", "text": "10 days in continuity"},
    ]
    docs = ["Fundamental & Subsidiary Rule & Leave Rule PPT.pdf", "Leave Rule PPT.pdf"]
    parts = ["14 days in a calendar year.", "10 days in continuity"]
    assert part_found(parts[1], chunks, docs)
    assert not part_found(parts[0], chunks, docs)
    metrics = stage_metrics(parts, chunks, docs)
    assert metrics["hit"] is True
    assert metrics["recall"] == 0.5
    assert metrics["source_hit"] is True
    assert metrics["mrr"] == 0.5


def test_golden_files_split_multi_quotes():
    full = load_items("full")
    sample = load_items("sample25")
    assert len(sample) == 25
    assert len(full) >= 200
    multi = next(item for item in full if item["id"] == "G163")
    assert len(multi["evidence_parts"]) == 2
    assert len(multi["source_docs"]) == 2
    assert multi["answerable"] is True
    refused = next(item for item in full if item["id"] == "G199")
    assert refused["answerable"] is False
    assert refused["evidence_parts"] == []


def test_gate_flags_only_past_the_threshold():
    summary = {
        "retrieval": {"recall_at_8": 0.80},
        "answer": {
            "faithfulness": 0.90,
            "false_refusal_rate": 0.05,
            "correct_refusal_rate": 0.90,
            "latency_p95_s": 15,
            "error_rate": 0.02,
        },
    }
    baseline = {"retrieval": {"recall_at_8": 0.83}, "answer": {}}
    limits = {
        "recall_at_8_drop_pts": 3,
        "faithfulness_min": 0.90,
        "false_refusal_max": 0.05,
        "correct_refusal_min": 0.90,
        "p95_latency_s_max": 15,
        "error_rate_max": 0.02,
    }
    assert gate_failures(summary, baseline, limits) == []

    summary["retrieval"]["recall_at_8"] = 0.79
    summary["answer"]["faithfulness"] = 0.89
    summary["answer"]["false_refusal_rate"] = 0.06
    summary["answer"]["correct_refusal_rate"] = 0.89
    summary["answer"]["latency_p95_s"] = 15.1
    summary["answer"]["error_rate"] = 0.03
    failures = gate_failures(summary, baseline, limits)
    assert len(failures) == 6
    assert gate_failures({"retrieval": {}, "answer": {}}, baseline, limits) == []


def test_judge_payload_excludes_grader_quotes():
    item = {
        "id": "G001",
        "question": "How much earnest money?",
        "language": "en",
        "expected_answer": "2 percent",
        "must_include_facts": ["2 percent"],
        "evidence_parts": ["SECRET-QUOTE-NOT-FOR-THE-MODEL"],
        "source_docs": ["Nivida Path.pdf"],
    }
    payload = judge_user_payload(item, "2 percent", "retrieved chunk")
    assert "SECRET-QUOTE-NOT-FOR-THE-MODEL" not in payload
    assert "Nivida Path.pdf" not in payload
    body = json.loads(payload)
    assert body["retrieved_context"] == "retrieved chunk"
    assert "evidence" not in body


def test_cache_key_changes_with_the_answer_and_estimate_scales():
    item = {
        "id": "G001",
        "question": "q",
        "expected_answer": "a",
        "must_include_facts": ["2"],
        "answerable": True,
        "type": "numeric",
    }
    assert cache_key(item, "one", "google/gemini-2.5-flash") != cache_key(
        item, "two", "google/gemini-2.5-flash"
    )
    assert estimate_cost(0, "google/gemini-2.5-flash") == 0
    assert estimate_cost(2, "google/gemini-2.5-flash") == 2 * estimate_cost(
        1, "google/gemini-2.5-flash"
    )


def test_history_header_and_append_keeps_both_rows(tmp_path):
    import csv

    from eval.history import COLUMNS, append_history_row

    path = tmp_path / "history.csv"
    first = ["run-a"] + [""] * (len(COLUMNS) - 1)
    second = ["run-b"] + [""] * (len(COLUMNS) - 1)
    append_history_row(first, path)
    append_history_row(second, path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == COLUMNS
    assert [row[0] for row in rows[1:]] == ["run-a", "run-b"]
    assert len(rows) == 3
