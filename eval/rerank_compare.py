"""Compare rerankers on one shared retrieval pool.

Retrieval runs once. Each backend only reranks that pool of 20. Suggested
thresholds are reported and not applied.
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

NAMES = ("minilm", "mmarco", "gte", "colbert", "none")


def clone_hit(hit):
    from vectorDB.rerankers import fused_rrf_score

    payload = dict(getattr(hit, "payload", None) or {})
    return SimpleNamespace(
        payload=payload,
        score=getattr(hit, "score", None),
        rrf_score=fused_rrf_score(hit),
    )


def relevant_scores(item: dict, reranked: list[dict]) -> list[float]:
    from eval.scoring import part_found

    if not item.get("answerable"):
        return []
    found = []
    for chunk in reranked:
        if chunk.get("score") is None:
            continue
        if any(
            part_found(part, [chunk], item.get("source_docs") or [])
            for part in (item.get("evidence_parts") or [])
        ):
            found.append(float(chunk["score"]))
    return found


def evaluate_backend(name: str, items: list[dict], pools: list[dict], rerank_fn, threshold) -> dict:
    """Score one backend over already-retrieved pools. rerank_fn(query, docs, top_k)."""
    from eval.retrieval_eval import _chunk, grade_retrieval
    from eval.run_eval import percentile, retrieval_summary, slices
    from vectorDB.rerankers import suggest_threshold

    rows = []
    latencies = []
    records = []
    for item, pool in zip(items, pools):
        clones = [clone_hit(hit) for hit in pool["merged_hits"]]
        started = time.perf_counter()
        pairs = rerank_fn(item["question"], clones, pool["top_k"])
        latencies.append((time.perf_counter() - started) * 1000.0)
        reranked = [_chunk(doc, float(score)) for doc, score in pairs]
        if threshold is None:
            kept = list(reranked)
        else:
            kept = [chunk for chunk in reranked if float(chunk["score"] or 0) >= threshold]
        top_score = None
        if reranked:
            top_score = max(float(chunk["score"] or 0) for chunk in reranked)
        retrieved = {
            "rrf_by_phrasing": pool["rrf_by_phrasing"],
            "merged": pool["merged"],
            "reranked": reranked,
            "kept": kept,
            "top_rerank_score": top_score,
        }
        row = {
            "id": item["id"],
            "language": item["language"],
            "type": item["type"],
            "difficulty": item["difficulty"],
            "answerable": item["answerable"],
            "top_rerank_score": top_score,
            "error": None,
        }
        if item["answerable"]:
            row.update(grade_retrieval(item, retrieved))
        records.append(
            {
                "answerable": bool(item["answerable"]),
                "top_score": top_score,
                "relevant_scores": relevant_scores(item, reranked),
            }
        )
        rows.append(row)

    summary = retrieval_summary(rows)
    grouped = slices(rows)

    def _scores(answerable: bool) -> list[float]:
        return [
            float(row["top_rerank_score"])
            for row in rows
            if bool(row.get("answerable")) is answerable and row.get("top_rerank_score") is not None
        ]

    answerable_scores = _scores(True)
    unanswerable_scores = _scores(False)
    return {
        "name": name,
        "rows": rows,
        "retrieval": summary,
        "slices": grouped,
        "hit": (summary.get("stages") or {}).get("reranked", {}).get("hit"),
        "recall": summary.get("recall_at_8"),
        "mrr": (summary.get("stages") or {}).get("reranked", {}).get("mrr"),
        "dropped": summary.get("dropped_by_reranker"),
        "killed": summary.get("killed_by_threshold"),
        "applied_threshold": threshold,
        "suggested_threshold": suggest_threshold(records),
        "answerable_p10": percentile(answerable_scores, 0.10),
        "answerable_p50": percentile(answerable_scores, 0.50),
        "answerable_p90": percentile(answerable_scores, 0.90),
        "unanswerable_p10": percentile(unanswerable_scores, 0.10),
        "unanswerable_p50": percentile(unanswerable_scores, 0.50),
        "unanswerable_p90": percentile(unanswerable_scores, 0.90),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "n_items": len(rows),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_ms(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f}"


def render_compare(reports: list[dict]) -> str:
    from eval.history import TYPE_COLUMNS

    headers = [
        "reranker",
        "hit@8",
        "recall@8",
        "MRR",
        "recall@8 en",
        "recall@8 hi",
    ]
    headers.extend(column.removeprefix("recall_at_8_") for column in TYPE_COLUMNS.values())
    headers.extend(
        [
            "reranker drop",
            "threshold kill",
            "ans p10",
            "ans p50",
            "ans p90",
            "unans p10",
            "unans p50",
            "unans p90",
            "rerank p50 ms",
            "rerank p95 ms",
            "applied threshold",
            "suggested threshold",
        ]
    )

    def cell_row(report: dict) -> list[str]:
        grouped = report.get("slices") or {}
        language = grouped.get("language") or {}
        types = grouped.get("type") or {}
        cells = [
            report["name"],
            _fmt(report.get("hit")),
            _fmt(report.get("recall")),
            _fmt(report.get("mrr")),
            _fmt((language.get("en") or {}).get("recall_at_8")),
            _fmt((language.get("hi") or {}).get("recall_at_8")),
        ]
        for type_name in TYPE_COLUMNS:
            cells.append(_fmt((types.get(type_name) or {}).get("recall_at_8")))
        cells.extend(
            [
                "" if report.get("dropped") is None else str(int(report["dropped"])),
                "" if report.get("killed") is None else str(int(report["killed"])),
                _fmt(report.get("answerable_p10"), 3),
                _fmt(report.get("answerable_p50"), 3),
                _fmt(report.get("answerable_p90"), 3),
                _fmt(report.get("unanswerable_p10"), 3),
                _fmt(report.get("unanswerable_p50"), 3),
                _fmt(report.get("unanswerable_p90"), 3),
                _fmt_ms(report.get("latency_p50_ms")),
                _fmt_ms(report.get("latency_p95_ms")),
                "skipped" if report.get("applied_threshold") is None else _fmt(report.get("applied_threshold"), 3),
                _fmt(report.get("suggested_threshold"), 3),
            ]
        )
        return cells

    lines = [
        "# Reranker comparison",
        "",
        "One retrieval pool per item, then each backend reranks that pool.",
        "Suggested thresholds maximize correct refusals plus kept answerable items on this golden set.",
        "They are not applied.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for report in reports:
        lines.append("| " + " | ".join(cell_row(report)) + " |")
    lines.append("")
    return "\n".join(lines)


def _activate(name: str):
    from vectorDB import search
    from vectorDB.rerankers import load_reranker

    if getattr(search.reranker, "name", None) == name:
        return search.reranker
    search.reranker = None
    gc.collect()
    os.environ["RERANKER"] = name
    search.reranker = load_reranker(name)
    return search.reranker


async def run_compare(args) -> int:
    from eval.gate import gate_failures, load_thresholds
    from eval.history import (
        HISTORY,
        append_history_row,
        dataset_version,
        history_row,
        ist_now,
        url_kind,
    )
    from eval.retrieval_eval import collect_pool
    from eval.run_eval import git_branch, git_sha, load_baseline
    from eval.scoring import load_items
    from vectorDB.rerankers import log_skipped_threshold, reject_threshold
    from vectorDB.search import init_services

    names = list(NAMES) if args.reranker == "all" else [args.reranker]
    which = args.retrieval_set or args.set
    os.environ["RERANKER"] = names[0]
    await asyncio.to_thread(init_services)
    from eval.run_eval import apply_limit

    items = apply_limit(load_items(which), getattr(args, "limit", None))
    timeout = float(getattr(args, "item_timeout", 60.0) or 60.0)
    pools = []
    item_errors: dict[str, str] = {}
    total = len(items)
    for index, item in enumerate(items, start=1):
        started = time.perf_counter()
        try:
            pools.append(await asyncio.wait_for(collect_pool(item["question"]), timeout))
            state = "ok"
        except Exception as exc:
            from eval.retrieval_eval import describe_failure

            pools.append(
                {
                    "phrasings": [],
                    "rrf_by_phrasing": [],
                    "merged": [],
                    "merged_hits": [],
                    "top_k": 8,
                    "rerank_pool": 20,
                    "rrf_candidates": 28,
                }
            )
            item_errors[item["id"]] = describe_failure(exc, timeout)
            state = f"error {item_errors[item['id']]}"
        print(f"retrieval {index}/{total} {item['id']} {time.perf_counter() - started:.1f}s {state}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "eval" / "results" / f"{stamp}_rerank_compare"
    out.mkdir(parents=True, exist_ok=True)
    try:
        rel_out = str(out.relative_to(ROOT))
    except ValueError:
        rel_out = str(out)

    reports = []
    published: list[tuple[dict, list]] = []
    exit_code = 0
    interrupted = False
    try:
        for name in names:
            backend = _activate(name)
            threshold = reject_threshold(name)
            if threshold is None:
                log_skipped_threshold()
            print(f"reranking with {name}")
            report = evaluate_backend(name, items, pools, backend.rerank, threshold)
            suggested = report["suggested_threshold"]
            shown = "none" if suggested is None else f"{suggested:.4f}"
            print(f"suggested threshold for {name}: {shown} (not applied)")
            summary = {
                "git_sha": git_sha(),
                "n_items": report["n_items"],
                "retrieval_set": which,
                "answer_set": None,
                "retrieval": report["retrieval"],
                "answer": None,
                "slices": report["slices"],
                "reranker": name,
            }
            baseline = load_baseline()
            failures = gate_failures(summary, baseline, load_thresholds())
            summary["gate_result"] = failures
            if args.gate and failures:
                exit_code = 1
                for failure in failures:
                    print(f"GATE {name} {failure}")
            summary["n_errors"] = len(item_errors)
            summary["history_meta"] = {
                "run_id": uuid.uuid4().hex[:12],
                "timestamp_ist": ist_now(),
                "git_branch": git_branch(),
                "pr_number": os.getenv("PR_NUMBER") or "",
                "cloud_run_revision": os.getenv("EVAL_REVISION") or "",
                "target_url_kind": url_kind(getattr(args, "url", None)),
                "dataset_version": dataset_version(),
                "judge_prompt_version": "",
                "n_refusal": None,
                "results_dir": rel_out,
                "reranker": name,
            }
            append_history_row(history_row(summary))
            print(f"appended {HISTORY} reranker={name}")
            for row in report["rows"]:
                message = item_errors.get(row["id"])
                if message:
                    row["status"] = "error"
                    row["error"] = message
            reports.append(report)
            published.append((summary, report["rows"]))
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        raise
    finally:
        if reports:
            text = render_compare(reports)
            (out / "compare.md").write_text(text, encoding="utf-8")
            print(text)
            print(f"wrote {out / 'compare.md'}")
        if published and not interrupted:
            from eval.upload import publish

            publish(published, out, no_upload=getattr(args, "no_upload", False))
    return exit_code
