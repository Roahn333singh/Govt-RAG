#!/usr/bin/env python3
"""Run retrieval and answer evals and write a report directory.

Does not import the FastAPI app and does not change serving behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.gate import gate_failures, load_thresholds  # noqa: E402
from eval.history import (  # noqa: E402
    HISTORY,
    append_history_row,
    backfill_history,
    dataset_version,
    history_row,
    ist_now,
    url_kind,
)
from eval.scoring import load_items  # noqa: E402

BASELINE = ROOT / "eval" / "baseline.json"
SLICE_KEYS = ("type", "language", "difficulty", "answerable")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    if os.getenv("GITHUB_HEAD_REF"):
        return os.environ["GITHUB_HEAD_REF"]
    if os.getenv("GITHUB_REF_NAME"):
        return os.environ["GITHUB_REF_NAME"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def percentile(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _mean(values: list[float]):
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _rate(flags: list[bool]):
    if not flags:
        return None
    return sum(1 for flag in flags if flag) / len(flags)


def retrieval_summary(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row.get("answerable")]
    recall = _mean([row.get("recall_at_8") for row in answerable])
    stages = {}
    for name in ("rrf_candidates", "merged", "reranked", "kept"):
        stages[name] = {
            "hit": _mean([(row.get("stages") or {}).get(name, {}).get("hit") for row in answerable]),
            "recall": _mean([(row.get("stages") or {}).get(name, {}).get("recall") for row in answerable]),
            "mrr": _mean([(row.get("stages") or {}).get(name, {}).get("mrr") for row in answerable]),
            "source_hit": _mean(
                [(row.get("stages") or {}).get(name, {}).get("source_hit") for row in answerable]
            ),
        }
    def score_stats(group: list[dict]) -> dict:
        scores = [row["top_rerank_score"] for row in group if row.get("top_rerank_score") is not None]
        return {
            "n": len(scores),
            "min": min(scores) if scores else None,
            "p50": percentile(scores, 0.50),
            "p90": percentile(scores, 0.90),
            "max": max(scores) if scores else None,
        }

    return {
        "n_answerable": len(answerable),
        "n_errors": sum(1 for row in rows if row.get("error")),
        "recall_at_8": recall,
        "stages": stages,
        "dropped_by_reranker": sum(1 for row in answerable if row.get("dropped_by_reranker")),
        "killed_by_threshold": sum(1 for row in answerable if row.get("killed_by_threshold")),
        "top_score": {
            "answerable": score_stats([row for row in rows if row.get("answerable")]),
            "unanswerable": score_stats([row for row in rows if not row.get("answerable")]),
        },
    }


def answer_summary(rows: list[dict], meta: dict | None) -> dict | None:
    if meta is None:
        return None
    judged = [row for row in rows if row.get("judge")]
    ok = [row for row in rows if not row.get("error")]
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]

    def refusal_rate(answerable: bool):
        group = [row for row in judged if bool(row.get("answerable")) is answerable]
        if not group:
            return None
        refused = [bool(row["judge"]["refusal"]) for row in group]
        if answerable:
            return _rate(refused)
        return _rate(refused)

    adversarial = [row for row in judged if row.get("type") == "adversarial"]
    injection_pass = None
    if adversarial:
        injection_pass = _rate([not row["judge"]["injection_followed"] for row in adversarial])
    return {
        "n": len(rows),
        "n_judged": len(judged),
        "correctness_mean": _mean([row["judge"]["correctness"] for row in judged]),
        "completeness_mean": _mean([row["judge"]["completeness"] for row in judged]),
        "faithfulness": _mean([row["judge"]["faithfulness"] for row in judged]),
        "language_match_rate": _mean(
            [1.0 if row["judge"]["language_match"] else 0.0 for row in judged]
        ),
        "correct_refusal_rate": refusal_rate(False),
        "false_refusal_rate": refusal_rate(True),
        "adversarial_injection_pass_rate": injection_pass,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "error_rate": (len(rows) - len(ok)) / len(rows) if rows else None,
        "cost_usd": meta.get("cost_usd"),
        "judge_model": meta.get("judge_model"),
        "judge_calls": meta.get("judge_calls"),
        "cached_judge_calls": meta.get("cached_judge_calls"),
        "stopped_for_cost": meta.get("stopped_for_cost"),
    }


def _slice_row(rows: list[dict]) -> dict:
    retrieval = retrieval_summary(rows) if any("top_rerank_score" in row or row.get("stages") for row in rows) else {}
    judged = [row for row in rows if row.get("judge")]
    out = {"n": len(rows)}
    if retrieval:
        out["recall_at_8"] = retrieval.get("recall_at_8")
        stages = retrieval.get("stages") or {}
        out["hit_at_8"] = (stages.get("reranked") or {}).get("hit")
        out["mrr"] = (stages.get("reranked") or {}).get("mrr")
        out["dropped_by_reranker"] = retrieval.get("dropped_by_reranker")
        out["killed_by_threshold"] = retrieval.get("killed_by_threshold")
        out["top_score"] = retrieval.get("top_score")
    if judged:
        answer = answer_summary(rows, {"cost_usd": None})
        out["faithfulness"] = answer["faithfulness"]
        out["false_refusal_rate"] = answer["false_refusal_rate"]
        out["correct_refusal_rate"] = answer["correct_refusal_rate"]
        out["correctness_mean"] = answer["correctness_mean"]
    return out


def slices(rows: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    for key in SLICE_KEYS:
        buckets: dict[str, list] = {}
        for row in rows:
            buckets.setdefault(str(row.get(key)), []).append(row)
        grouped[key] = {name: _slice_row(bucket) for name, bucket in sorted(buckets.items())}
    return grouped


def load_baseline() -> dict:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def deltas(summary: dict, baseline: dict) -> dict:
    pairs = {
        "recall_at_8": (
            (summary.get("retrieval") or {}).get("recall_at_8"),
            (baseline.get("retrieval") or {}).get("recall_at_8"),
        ),
        "faithfulness": (
            (summary.get("answer") or {}).get("faithfulness"),
            (baseline.get("answer") or {}).get("faithfulness"),
        ),
        "false_refusal_rate": (
            (summary.get("answer") or {}).get("false_refusal_rate"),
            (baseline.get("answer") or {}).get("false_refusal_rate"),
        ),
        "correct_refusal_rate": (
            (summary.get("answer") or {}).get("correct_refusal_rate"),
            (baseline.get("answer") or {}).get("correct_refusal_rate"),
        ),
        "latency_p95_s": (
            (summary.get("answer") or {}).get("latency_p95_s"),
            (baseline.get("answer") or {}).get("latency_p95_s"),
        ),
        "error_rate": (
            (summary.get("answer") or {}).get("error_rate"),
            (baseline.get("answer") or {}).get("error_rate"),
        ),
    }
    out = {}
    for name, (current, base) in pairs.items():
        if current is None or base is None:
            out[name] = None
        else:
            out[name] = float(current) - float(base)
    return out


def _fmt(value, kind: str = "num") -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{float(value):.1%}"
    if kind == "pts":
        return f"{float(value) * 100:+.1f} pts"
    if kind == "sec":
        return f"{float(value):.2f}s"
    if kind == "usd":
        return f"${float(value):.4f}"
    return f"{float(value):.3f}"


def render_report(summary: dict) -> str:
    retrieval = summary.get("retrieval") or {}
    answer = summary.get("answer") or {}
    delta = summary.get("deltas") or {}
    lines = [
        "# Eval report",
        "",
        f"- git `{summary.get('git_sha')}`",
        f"- items {summary.get('n_items')}",
        "",
        "## Retrieval",
        "",
        f"- recall@8 {_fmt(retrieval.get('recall_at_8'))} ({_fmt(delta.get('recall_at_8'), 'pts')} vs baseline)",
        f"- reranker dropped a relevant chunk on {retrieval.get('dropped_by_reranker', 0)} items",
        f"- reject threshold removed a relevant chunk on {retrieval.get('killed_by_threshold', 0)} items",
        "",
        "| stage | hit | recall | MRR | source hit |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in (retrieval.get("stages") or {}).items():
        lines.append(
            f"| {name} | {_fmt(metrics.get('hit'), 'pct')} | {_fmt(metrics.get('recall'), 'pct')} "
            f"| {_fmt(metrics.get('mrr'))} | {_fmt(metrics.get('source_hit'), 'pct')} |"
        )
    top = retrieval.get("top_score") or {}
    lines += ["", "Top rerank score (for REJECT_THRESHOLD):", ""]
    for label in ("answerable", "unanswerable"):
        stats = top.get(label) or {}
        lines.append(
            f"- {label}: n={stats.get('n', 0)} min={_fmt(stats.get('min'))} "
            f"p50={_fmt(stats.get('p50'))} p90={_fmt(stats.get('p90'))} max={_fmt(stats.get('max'))}"
        )
    lines += [
        "",
        "## Answers",
        "",
        f"- faithfulness {_fmt(answer.get('faithfulness'))} ({_fmt(delta.get('faithfulness'), 'pts')} vs baseline)",
        f"- completeness {_fmt(answer.get('completeness_mean'), 'pct')}",
        f"- correctness {_fmt(answer.get('correctness_mean'))} / 2",
        f"- language match {_fmt(answer.get('language_match_rate'), 'pct')}",
        f"- correct refusal {_fmt(answer.get('correct_refusal_rate'), 'pct')} ({_fmt(delta.get('correct_refusal_rate'), 'pts')})",
        f"- false refusal {_fmt(answer.get('false_refusal_rate'), 'pct')} ({_fmt(delta.get('false_refusal_rate'), 'pts')})",
        f"- adversarial injection pass {_fmt(answer.get('adversarial_injection_pass_rate'), 'pct')}",
        f"- latency p50 {_fmt(answer.get('latency_p50_s'), 'sec')} p95 {_fmt(answer.get('latency_p95_s'), 'sec')} ({_fmt(delta.get('latency_p95_s'), 'sec')} vs baseline)",
        f"- error rate {_fmt(answer.get('error_rate'), 'pct')}",
        f"- judge cost {_fmt(answer.get('cost_usd'), 'usd')} ({answer.get('judge_calls', 0)} calls, {answer.get('cached_judge_calls', 0)} cached, {answer.get('judge_model') or '—'})",
        "",
        "## Slices",
        "",
    ]
    for key, groups in (summary.get("slices") or {}).items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| slice | n | recall@8 | faithfulness | false refusal |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, metrics in groups.items():
            lines.append(
                f"| {name} | {metrics.get('n', 0)} | {_fmt(metrics.get('recall_at_8'), 'pct')} "
                f"| {_fmt(metrics.get('faithfulness'))} | {_fmt(metrics.get('false_refusal_rate'), 'pct')} |"
            )
        lines.append("")
    failures = summary.get("gate") or []
    lines.append("## Gate")
    lines.append("")
    if not failures:
        lines.append("No gate failures.")
    else:
        for failure in failures:
            lines.append(f"- {failure}")
    lines.append("")
    return "\n".join(lines)


def write_baseline(summary: dict) -> None:
    payload = {
        "git_sha": summary.get("git_sha"),
        "updated": summary.get("timestamp"),
        "retrieval": {"recall_at_8": (summary.get("retrieval") or {}).get("recall_at_8")},
        "answer": {
            "faithfulness": (summary.get("answer") or {}).get("faithfulness"),
            "false_refusal_rate": (summary.get("answer") or {}).get("false_refusal_rate"),
            "correct_refusal_rate": (summary.get("answer") or {}).get("correct_refusal_rate"),
            "latency_p95_s": (summary.get("answer") or {}).get("latency_p95_s"),
            "error_rate": (summary.get("answer") or {}).get("error_rate"),
        },
    }
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_limit(items: list, limit: int | None) -> list:
    """First N items. None keeps the whole set."""
    if limit is None:
        return list(items)
    return list(items)[: max(limit, 0)]


def count_errors(rows: list[dict]) -> int:
    total = 0
    for row in rows:
        if row.get("status") == "error" or row.get("answer_error"):
            total += 1
            continue
        err = row.get("error")
        if isinstance(err, str) and err:
            total += 1
    return total


def _public_row(row: dict) -> dict:
    """Drop grader-only fields if a caller attached them."""
    hidden = {"evidence_parts", "evidence_quote", "source_docs", "source_doc", "judge_key"}
    return {key: value for key, value in row.items() if key not in hidden}


async def _run(args) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    retrieval_set = args.retrieval_set or args.set
    answer_set = args.answer_set or args.set
    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    answer_meta = None

    if not args.answer_only:
        from eval.retrieval_eval import run_retrieval

        retrieval_rows = await run_retrieval(
            apply_limit(load_items(retrieval_set), args.limit),
            item_timeout=args.item_timeout,
        )
    if not args.retrieval_only:
        if not args.url:
            raise SystemExit("--url is required for the answer eval")
        from eval.answer_eval import run_answers

        result = await run_answers(
            apply_limit(load_items(answer_set), args.limit),
            args.url,
            max_cost=args.max_cost,
            model=args.judge_model,
            item_timeout=args.item_timeout,
        )
        answer_rows = result["items"]
        answer_meta = result

    by_id: dict[str, dict] = {}
    for row in retrieval_rows:
        by_id[row["id"]] = _public_row(row)
    for row in answer_rows:
        merged = by_id.setdefault(row["id"], {"id": row["id"]})
        for key in ("language", "type", "difficulty", "answerable"):
            merged.setdefault(key, row.get(key))
        for key in ("status", "latency_s", "answer", "sources", "cached", "judge", "judge_cached"):
            if key in row:
                merged[key] = row[key]
        if row.get("error"):
            merged["answer_error"] = row["error"]
        elif "error" not in merged:
            merged["error"] = None

    rows = list(by_id.values())
    summary = {
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "n_items": len(rows),
        "retrieval_set": None if args.answer_only else retrieval_set,
        "answer_set": None if args.retrieval_only else answer_set,
        "retrieval": retrieval_summary(retrieval_rows) if retrieval_rows else {},
        "answer": answer_summary(answer_rows, answer_meta) if answer_rows else None,
        "slices": slices(rows),
        "n_errors": count_errors(rows),
    }
    baseline = load_baseline()
    summary["deltas"] = deltas(summary, baseline)
    gate_result = gate_failures(summary, baseline, load_thresholds())
    summary["gate_result"] = gate_result
    summary["gate"] = gate_result if args.gate else []

    out = Path(args.out) if args.out else ROOT / "eval" / "results" / f"{summary['timestamp']}_{summary['git_sha']}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_public_row(row), ensure_ascii=False) + "\n")
    from vectorDB.rerankers import selected_name

    judged = [row for row in answer_rows if row.get("judge")]
    n_refusal = sum(1 for row in judged if row["judge"].get("refusal")) if judged else None
    try:
        rel_out = str(out.relative_to(ROOT))
    except ValueError:
        rel_out = str(out)
    variant = (args.variant or os.getenv("EVAL_VARIANT") or "").strip() or None
    summary["variant"] = variant
    if summary.get("retrieval"):
        top = summary["retrieval"].get("top_score") or {}
        print(
            "top_rerank_score answerable="
            f"{top.get('answerable')} unanswerable={top.get('unanswerable')}"
        )
    if args.compare_to:
        from eval.stats import paired_against

        p_values = {}
        for raw in args.compare_to:
            if "=" in raw:
                label, path = raw.split("=", 1)
            else:
                label, path = Path(raw).name, raw
            p_values[label] = paired_against(out, Path(path), label)
        summary["p_values"] = p_values
    summary["history_meta"] = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp_ist": ist_now(),
        "git_branch": git_branch(),
        "pr_number": os.getenv("PR_NUMBER") or "",
        "cloud_run_revision": os.getenv("EVAL_REVISION") or "",
        "target_url_kind": url_kind(args.url),
        "dataset_version": dataset_version(),
        "judge_prompt_version": "" if answer_meta is None else "answer-judge-v1",
        "n_refusal": n_refusal,
        "results_dir": rel_out,
        "reranker": selected_name(),
        "variant": variant or "",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_history_row(history_row(summary))
    print(f"appended {HISTORY}")
    report = render_report(summary)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}")
    from eval.upload import publish

    publish([(summary, rows)], out, no_upload=args.no_upload)

    if args.update_baseline:
        write_baseline(summary)
        print(f"updated {BASELINE}")
    if answer_meta and answer_meta.get("stopped_for_cost"):
        return 2
    if summary["gate"]:
        for failure in summary["gate"]:
            print(f"GATE {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-set eval and write a report")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Rebuild history.csv from eval/results/*/summary.json",
    )
    parser.add_argument("--set", choices=("full", "sample25", "prod50"), default="full")
    parser.add_argument("--retrieval-set", choices=("full", "sample25", "prod50"))
    parser.add_argument("--answer-set", choices=("full", "sample25", "prod50"))
    parser.add_argument("--url", help="Cloud Run origin for POST /chat")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--max-cost", type=float, default=0.30)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--item-timeout", type=float, default=60.0, help="Seconds allowed for one retrieval or /chat item")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N items of each chosen set")
    parser.add_argument(
        "--variant",
        default=None,
        help="Tag for summary.json / history (e.g. colbert-jina-v2).",
    )
    parser.add_argument(
        "--compare-to",
        action="append",
        default=[],
        help="Paired p-values vs another results dir. label=path or path. Repeatable.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the Google Cloud Storage upload. Pull requests skip it either way.",
    )
    parser.add_argument(
        "--reranker",
        choices=("minilm", "mmarco", "gte", "colbert", "none", "all"),
        help="Retrieval-only comparison. 'all' reranks one shared pool with each backend.",
    )
    args = parser.parse_args()
    if args.backfill:
        count = backfill_history()
        print(f"backfilled {count} rows to {HISTORY}")
        return
    if args.reranker:
        from eval.rerank_compare import run_compare

        raise SystemExit(asyncio.run(run_compare(args)))
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
