#!/usr/bin/env python3
"""Production-style load test against /chat using the QA API key.

Reads:
  LOADTEST_URL or --url     Cloud Run origin (candidate / loadtest tag)
  RAG_EVAL_API_KEY or QA_API_KEY or LOADTEST_API_KEY
  Optional --upload to gs://govt-rag-evals/load/

Never prints the API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROD50 = ROOT / "eval" / "prod50.jsonl"
BUCKET = "govt-rag-evals"
PROJECT = "project-0ff01dde-a087-4c96-b04"

TIMING_KEYS = (
    "embed_ms",
    "qdrant_ms",
    "rerank_ms",
    "llm_first_token_ms",
    "llm_total_ms",
)

# (name, concurrency, duration_s)
DEFAULT_STAGES = [
    ("ramp_1", 1, 180),
    ("ramp_5", 5, 180),
    ("ramp_10", 10, 180),
    ("ramp_20", 20, 180),
    ("ramp_40", 40, 180),
    ("spike_30", 30, 60),
    ("soak_5", 5, 1200),
]

# Single-instance capacity sweep. 20s cooldown between steps is applied in _amain.
CAP_STAGES = [
    ("cap_1", 1, 120),
    ("cap_2", 2, 120),
    ("cap_3", 3, 120),
    ("cap_4", 4, 120),
    ("cap_6", 6, 120),
    ("cap_8", 8, 120),
]

CAP_COOLDOWN_S = 20.0
CAP_P95_LIMIT_S = 12.0


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (rank - low), 3)


def _load_questions(limit: int | None = None) -> list[dict]:
    items = []
    with PROD50.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(row)
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise SystemExit(f"no questions in {PROD50}")
    return items


def _api_key() -> str:
    key = (
        os.getenv("RAG_EVAL_API_KEY")
        or os.getenv("QA_API_KEY")
        or os.getenv("LOADTEST_API_KEY")
        or ""
    ).strip()
    if not key:
        raise SystemExit(
            "Set RAG_EVAL_API_KEY, QA_API_KEY, or LOADTEST_API_KEY (value never printed)"
        )
    return key


def _timing_summary(rows: list[dict]) -> dict:
    out: dict[str, dict[str, float | None]] = {}
    for key in TIMING_KEYS:
        values = []
        for row in rows:
            timings = row.get("timings") or {}
            if not isinstance(timings, dict):
                continue
            raw = timings.get(key)
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        out[key] = {
            "p50_ms": _pct(values, 0.50),
            "p95_ms": _pct(values, 0.95),
            "n": len(values),
        }
    return out


def _summarize(rows: list[dict], *, wall_s: float, concurrency: int) -> dict:
    latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
    statuses = [r.get("status") for r in rows]
    n = len(rows) or 1
    summary = {
        "n_requests": len(rows),
        "concurrency": concurrency,
        "wall_s": round(wall_s, 2),
        "rps": round(len(rows) / wall_s, 3) if wall_s > 0 else 0.0,
        "latency_p50_s": _pct(latencies, 0.50),
        "latency_p95_s": _pct(latencies, 0.95),
        "latency_p99_s": _pct(latencies, 0.99),
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else None,
        "ok_200": sum(1 for s in statuses if s == 200),
        "http_429": sum(1 for s in statuses if s == 429),
        "http_5xx": sum(1 for s in statuses if isinstance(s, int) and s >= 500),
        "http_503": sum(1 for s in statuses if s == 503),
        "timeouts": sum(1 for r in rows if r.get("error") == "TimeoutException"),
        "errors": sum(
            1
            for r in rows
            if r.get("error") or (isinstance(r.get("status"), int) and r["status"] >= 400)
        ),
        "error_rate": round(
            sum(
                1
                for r in rows
                if r.get("error") or (isinstance(r.get("status"), int) and r["status"] >= 400)
            )
            / n,
            4,
        ),
        "timings": _timing_summary(rows),
    }
    return summary


def _parse_timings(response: httpx.Response, body: dict) -> dict | None:
    timings = body.get("timings")
    if isinstance(timings, dict):
        return timings
    raw = response.headers.get("x-timing")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _one(
    client: httpx.AsyncClient,
    *,
    url: str,
    key: str,
    run_id: str,
    item: dict,
    n: int,
    timeout: float,
) -> dict:
    question = item.get("question") or ""
    tagged = f"{question} (ref LT-{run_id}-{n})"
    headers = {
        "X-API-Key": key,
        "X-QA-Run-Id": run_id,
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    try:
        response = await client.post(
            url.rstrip("/") + "/chat",
            json={"message": tagged},
            headers=headers,
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError:
            body = {}
        return {
            "n": n,
            "id": item.get("id"),
            "language": item.get("language"),
            "status": response.status_code,
            "latency_s": round(latency, 3),
            "request_id": response.headers.get("x-request-id"),
            "cached": body.get("cached"),
            "error": None,
            "timings": _parse_timings(response, body if isinstance(body, dict) else {}),
        }
    except httpx.TimeoutException:
        return {
            "n": n,
            "id": item.get("id"),
            "language": item.get("language"),
            "status": None,
            "latency_s": round(time.perf_counter() - started, 3),
            "request_id": None,
            "cached": None,
            "error": "TimeoutException",
            "timings": None,
        }
    except Exception as exc:
        return {
            "n": n,
            "id": item.get("id"),
            "language": item.get("language"),
            "status": None,
            "latency_s": round(time.perf_counter() - started, 3),
            "request_id": None,
            "cached": None,
            "error": type(exc).__name__,
            "timings": None,
        }


async def _run_stage(
    *,
    name: str,
    concurrency: int,
    duration_s: float,
    url: str,
    key: str,
    run_id: str,
    items: list[dict],
    timeout: float,
    out_dir: Path,
) -> dict:
    rows: list[dict] = []
    counter = {"n": 0}
    end = time.monotonic() + duration_s
    limits = httpx.Limits(
        max_connections=max(concurrency + 8, 16),
        max_keepalive_connections=max(concurrency + 8, 16),
    )

    async def worker() -> None:
        async with httpx.AsyncClient(limits=limits, http2=False) as client:
            while time.monotonic() < end:
                n = counter["n"]
                counter["n"] = n + 1
                item = items[n % len(items)]
                row = await _one(
                    client,
                    url=url,
                    key=key,
                    run_id=run_id,
                    item=item,
                    n=n,
                    timeout=timeout,
                )
                row["stage"] = name
                rows.append(row)
                t = row.get("timings") or {}
                timing_bit = ""
                if t:
                    timing_bit = (
                        f" embed={t.get('embed_ms')} qdrant={t.get('qdrant_ms')} "
                        f"rerank={t.get('rerank_ms')} llm1={t.get('llm_first_token_ms')} "
                        f"llm={t.get('llm_total_ms')}"
                    )
                print(
                    f"{name} n={n} id={row.get('id')} status={row.get('status')} "
                    f"lat={row.get('latency_s')}s"
                    + (f" err={row['error']}" if row.get("error") else "")
                    + timing_bit,
                    flush=True,
                )

    wall0 = time.monotonic()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.monotonic() - wall0
    summary = _summarize(rows, wall_s=wall, concurrency=concurrency)
    summary["stage"] = name
    summary["duration_s"] = duration_s
    (out_dir / f"{name}.json").write_text(
        json.dumps({"summary": summary, "requests": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _upload(out_dir: Path, run_id: str, timestamp_ist: str, stages: list[dict]) -> None:
    try:
        from google.cloud import storage
    except ImportError:
        print("warning: google-cloud-storage missing; skip upload")
        return
    try:
        client = storage.Client(project=PROJECT)
        bucket = client.bucket(BUCKET)
        payload = {
            "run_id": run_id,
            "timestamp_ist": timestamp_ist,
            "path": f"load/{run_id}/",
            "stages": stages,
        }
        history_blob = bucket.blob("load/history.json")
        try:
            history = json.loads(history_blob.download_as_text())
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
        history.append(
            {
                "run_id": run_id,
                "timestamp_ist": timestamp_ist,
                "stages": [
                    {
                        k: s.get(k)
                        for k in (
                            "stage",
                            "concurrency",
                            "n_requests",
                            "rps",
                            "latency_p50_s",
                            "latency_p95_s",
                            "latency_p99_s",
                            "error_rate",
                            "http_5xx",
                            "http_503",
                            "timeouts",
                            "timings",
                        )
                    }
                    for s in stages
                ],
            }
        )
        bucket.blob(f"load/{run_id}/summary.json").upload_from_string(
            json.dumps(payload, indent=2), content_type="application/json"
        )
        for path in out_dir.glob("*.json"):
            bucket.blob(f"load/{run_id}/{path.name}").upload_from_filename(
                str(path), content_type="application/json"
            )
        history_blob.upload_from_string(
            json.dumps(history, indent=2), content_type="application/json"
        )
        bucket.blob("load/latest.json").upload_from_string(
            json.dumps(
                {"run_id": run_id, "timestamp_ist": timestamp_ist, "path": f"load/{run_id}/"},
                indent=2,
            ),
            content_type="application/json",
        )
        print(f"uploaded gs://{BUCKET}/load/{run_id}/")
    except Exception as exc:
        print(f"warning: load upload failed ({type(exc).__name__})")


def _print_cap_table(summaries: list[dict]) -> int | None:
    """Print capacity table; return highest concurrency with p95 < 12s and 0 errors."""
    header = (
        f"{'stage':8} {'c':>3} {'n':>4} {'rps':>6} {'p50':>6} {'p95':>6} {'err':>4} "
        f"{'emb50':>7} {'qdr50':>7} {'rrk50':>7} {'ttft50':>7} {'llm50':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    capacity: int | None = None
    for s in summaries:
        t = s.get("timings") or {}

        def p50(key: str) -> str:
            val = (t.get(key) or {}).get("p50_ms")
            return f"{val:.0f}" if isinstance(val, (int, float)) else "-"

        p50_lat = s.get("latency_p50_s")
        p95_lat = s.get("latency_p95_s")
        print(
            f"{s.get('stage', ''):8} {s.get('concurrency', 0):3d} {s.get('n_requests', 0):4d} "
            f"{s.get('rps', 0):6.3f} "
            f"{(p50_lat if p50_lat is not None else 0):6.2f} "
            f"{(p95_lat if p95_lat is not None else 0):6.2f} "
            f"{s.get('errors', 0):4d} "
            f"{p50('embed_ms'):>7} {p50('qdrant_ms'):>7} {p50('rerank_ms'):>7} "
            f"{p50('llm_first_token_ms'):>7} {p50('llm_total_ms'):>7}",
            flush=True,
        )
        p95 = s.get("latency_p95_s")
        if (
            isinstance(p95, (int, float))
            and p95 < CAP_P95_LIMIT_S
            and int(s.get("errors") or 0) == 0
        ):
            capacity = int(s.get("concurrency") or 0)
    if capacity is None:
        print("capacity = none", flush=True)
    else:
        print(f"capacity = {capacity}", flush=True)
    return capacity


async def _preflight(url: str, key: str, timeout: float) -> None:
    """Fail fast if the origin/tag is missing (Cloud Run returns 404 for unknown tags)."""
    origin = url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(origin + "/health")
        if health.status_code != 200:
            raise SystemExit(
                f"preflight GET {origin}/health -> {health.status_code}. "
                "That host is not serving this service (often a missing Cloud Run "
                "--tag). Deploy with --no-traffic --tag=loadtest, then use the "
                "tagged URL from `gcloud run services describe`."
            )
        probe = await client.post(
            origin + "/chat",
            json={"message": "loadtest preflight"},
            headers={
                "X-API-Key": key,
                "X-QA-Run-Id": "preflight",
                "Content-Type": "application/json",
            },
        )
        if probe.status_code == 404:
            raise SystemExit(f"preflight POST {origin}/chat -> 404 (route missing)")
        if probe.status_code == 401:
            raise SystemExit(
                "preflight POST /chat -> 401 (API key rejected; check RAG_EVAL_API_KEY)"
            )
        if probe.status_code >= 500:
            raise SystemExit(f"preflight POST /chat -> {probe.status_code}")
        print(f"preflight ok health=200 chat={probe.status_code}", flush=True)


async def _amain(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    url = (args.url or os.getenv("LOADTEST_URL") or "").strip()
    if not url:
        raise SystemExit("Pass --url or set LOADTEST_URL")
    key = _api_key()
    items = _load_questions(args.limit)
    run_id = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "eval" / "results" / f"load_{stamp}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stages_spec = list(DEFAULT_STAGES)
    if args.stages == "smoke":
        stages_spec = [("ramp_1", 1, 30), ("ramp_5", 5, 30), ("spike_10", 10, 20)]
    elif args.stages == "cap":
        stages_spec = list(CAP_STAGES)
    elif args.skip_soak:
        stages_spec = [s for s in stages_spec if not s[0].startswith("soak")]

    print(
        f"loadtest run_id={run_id} url={url} questions={len(items)} "
        f"stages={len(stages_spec)} out={out_dir}",
        flush=True,
    )
    if not args.dry_run:
        await _preflight(url, key, args.timeout)
    summaries = []
    for index, (name, concurrency, duration) in enumerate(stages_spec):
        if args.dry_run:
            print(f"dry-run would run {name} c={concurrency} for {duration}s")
            continue
        summary = await _run_stage(
            name=name,
            concurrency=concurrency,
            duration_s=float(duration),
            url=url,
            key=key,
            run_id=run_id,
            items=items,
            timeout=args.timeout,
            out_dir=out_dir,
        )
        summaries.append(summary)
        if args.stages == "cap" and index < len(stages_spec) - 1:
            print(f"cooldown {CAP_COOLDOWN_S:.0f}s before next concurrency step", flush=True)
            await asyncio.sleep(CAP_COOLDOWN_S)

    capacity = None
    if args.stages == "cap" and summaries:
        capacity = _print_cap_table(summaries)

    meta = {
        "run_id": run_id,
        "timestamp_utc": stamp,
        "url": url,
        "stages": summaries,
    }
    if args.stages == "cap":
        meta["capacity"] = capacity
        meta["capacity_rule"] = f"highest concurrency with p95 < {CAP_P95_LIMIT_S}s and 0 errors"
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if args.upload and summaries:
        # IST-ish label for the dashboard without importing zoneinfo edge cases.
        ts_ist = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        _upload(out_dir, run_id, ts_ist, summaries)
    print(f"wrote {out_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="Cloud Run origin")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None, help="Use first N prod50 questions")
    parser.add_argument(
        "--stages",
        choices=("full", "smoke", "cap"),
        default="full",
        help="full ramp, short smoke, or single-instance capacity sweep",
    )
    parser.add_argument("--skip-soak", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Upload to gs://govt-rag-evals/load/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
