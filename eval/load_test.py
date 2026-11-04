"""Closed-loop step concurrency load test for /chat, with quality scoring.

Each level runs N virtual users for --duration seconds. Every user sends a
question, waits for the answer, then sends the next. Questions come from one
half of eval/load_questions.jsonl and each request gets a unique tag
"(ref LT-<run>-<n>)" so the exact cache never hits and the test entries can be
purged afterwards with: scripts/purge_cache.py --contains "LT-<run>".

Usage:
  read -s PROD_API_KEY && export PROD_API_KEY      # never printed
  uv run python eval/load_test.py --url URL --half A [--levels 1,2,4] [--duration 150] [--dry-run]
"""
from __future__ import annotations

import argparse, asyncio, json, os, statistics, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.prod50_test import is_refusal, score  # noqa: E402  same scoring as prod50

LEVELS = "1,2,4,8,12,16,20,30,40,60,100"
OK_LAT = 15.0


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, max(0, int(round(q * len(s) + 0.5)) - 1))], 3)


async def one(client, url, key, it, tag, timeout):
    q = f"{it['question']} (ref {tag})"
    t0 = time.perf_counter()
    try:
        r = await client.post(url.rstrip("/") + "/chat", json={"question": q},
                              headers={"X-API-Key": key}, timeout=timeout)
        lat = time.perf_counter() - t0
        try:
            d = r.json()
        except ValueError:
            d = {}
        res = {"status": r.status_code, "latency": lat, "rid": r.headers.get("X-Request-ID"),
               "cached": d.get("cached"), "answer": d.get("answer"), "sources": d.get("sources") or []}
    except Exception as e:
        res = {"status": None, "error": type(e).__name__, "latency": time.perf_counter() - t0,
               "rid": None, "cached": None, "answer": None, "sources": []}
    ok = res["status"] == 200
    ref = ok and is_refusal(res["answer"], res["sources"])
    rr = score(it, res["sources"])[0] if ok and not ref else 0.0
    return {"id": it["id"], "lang": it["language"], "question": q, "status": res["status"],
            "error": res.get("error"), "latency": round(res["latency"], 3), "cached": res["cached"],
            "rr": rr, "hit": rr > 0, "refusal": ref, "rid": res["rid"],
            "success": ok and res["latency"] < OK_LAT and rr > 0}


async def run_level(c, a, key, items, counter, run_id, log):
    rows, end = [], time.monotonic() + a.duration
    limits = httpx.Limits(max_connections=c + 5, max_keepalive_connections=c + 5)
    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        async def user():
            while time.monotonic() < end:
                n = counter[0]; counter[0] += 1
                it = items[n % len(items)]
                row = await one(client, a.url, key, it, f"LT-{run_id}-{n}", a.timeout)
                row["level"] = c; row["n"] = n
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n"); log.flush()
                print(f"c={c:<3} n={n:<5} {row['id']} status={row['status']} {row['latency']:.2f}s "
                      f"cached={row['cached']} rr={row['rr']:.2f}{' REFUSAL' if row['refusal'] else ''}"
                      f"{' ' + row['error'] if row['error'] else ''} rid={row['rid']}")
        t0 = time.monotonic()
        await asyncio.gather(*(user() for _ in range(c)))
        wall = time.monotonic() - t0
    return rows, wall


def summarize(c, rows, wall):
    n = len(rows)
    ok = [r for r in rows if r["status"] == 200]
    lat = [r["latency"] for r in ok]
    miss = [r for r in ok if r["cached"] is False]
    err = n - len(ok)
    return {"level": c, "requests": n, "wall_s": round(wall, 1),
            "ok_per_min": round(len(ok) / wall * 60, 1) if wall else None,
            "success_per_min": round(sum(r["success"] for r in rows) / wall * 60, 1) if wall else None,
            "p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99),
            "miss_p50": pct([r["latency"] for r in miss], 0.5), "miss_p95": pct([r["latency"] for r in miss], 0.95),
            "error_rate": round(err / n, 4) if n else None,
            "rate_429": round(sum(r["status"] == 429 for r in rows) / n, 4) if n else None,
            "rate_503": round(sum(r["status"] == 503 for r in rows) / n, 4) if n else None,
            "timeouts": sum(r["error"] is not None for r in rows),
            "cached_share": round(sum(bool(r["cached"]) for r in ok) / len(ok), 4) if ok else None,
            "mrr": round(sum(r["rr"] for r in ok) / len(ok), 4) if ok else None,
            "hit_rate": round(sum(r["hit"] for r in ok) / len(ok), 4) if ok else None,
            "refusal_rate": round(sum(r["refusal"] for r in ok) / len(ok), 4) if ok else None,
            "success_rate": round(sum(r["success"] for r in rows) / n, 4) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--half", choices=["A", "B", "all"], default="A", help="A for reranker, B for no reranker")
    ap.add_argument("--levels", default=LEVELS)
    ap.add_argument("--duration", type=float, default=150)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--max-error", type=float, default=0.05)
    ap.add_argument("--max-p95", type=float, default=30)
    ap.add_argument("--items", default=str(ROOT / "eval" / "load_questions.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    if a.half != "all":
        items = [i for i in items if i["half"] == a.half]
    run_id = uuid.uuid4().hex[:6]
    print(f"URL {a.url} | half {a.half} ({len(items)} questions) | levels {levels} | {a.duration:.0f}s each "
          f"| max {len(levels) * a.duration / 60:.1f} min | run id LT-{run_id}")
    print(f"stop rule: error rate > {a.max_error:.0%} or p95 > {a.max_p95:.0f}s | success = 200, < {OK_LAT:.0f}s, correct citation")
    if a.dry_run:
        for i, it in enumerate(items[:5]):
            print(f"  sample POST /chat {it['id']} [{it['language']}] {it['question'][:70]} (ref LT-{run_id}-{i})")
        sr = score(items[0], [{"source_document": items[0]["source_doc"], "content": items[0]["evidence_quote"],
                               "page_start": None}])
        print(f"  scoring self-check on {items[0]['id']}: rr={sr[0]} via {sr[1]} (expect 1.0 text)")
        print(f"  cleanup after: uv run python scripts/purge_cache.py --dry-run --contains \"LT-{run_id}\"")
        print("dry run: nothing sent")
        return
    key = os.getenv("PROD_API_KEY")
    if not key:
        sys.exit("PROD_API_KEY is not set")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "eval" / "results" / f"load_{stamp}{'_' + a.label if a.label else ''}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"url": a.url, "half": a.half, "levels": levels, "duration": a.duration, "timeout": a.timeout,
            "run_id": f"LT-{run_id}", "utc": stamp}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    try:
        print("warm-up /health", httpx.get(a.url.rstrip("/") + "/health", timeout=90).status_code)
    except Exception as e:
        print("warm-up failed:", type(e).__name__)
    summaries, counter = [], [0]
    with open(out / "requests.jsonl", "w", encoding="utf-8") as log:
        for c in levels:
            rows, wall = asyncio.run(run_level(c, a, key, items, counter, run_id, log))
            s = summarize(c, rows, wall)
            summaries.append(s)
            (out / "levels.json").write_text(json.dumps(summaries, indent=2))
            print(f"== level {c}: {json.dumps(s)}")
            if (s["error_rate"] or 0) > a.max_error or (s["p95"] or 0) > a.max_p95:
                print(f"stopping: level {c} broke the stop rule")
                break
    cap = [s["level"] for s in summaries if (s["success_rate"] or 0) >= 0.95]
    md = [f"# load test {stamp}", "", f"URL {a.url} | half {a.half} | run id LT-{run_id}", "",
          f"Highest level with >= 95% success: {max(cap) if cap else 'none'}", "",
          "| users | req | ok/min | success/min | p50 | p95 | p99 | miss p50 | err | 429 | 503 | cached | MRR | hit | success |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        md.append("| " + " | ".join(str(s[k]) for k in ("level", "requests", "ok_per_min", "success_per_min", "p50", "p95", "p99",
                  "miss_p50", "error_rate", "rate_429", "rate_503", "cached_share", "mrr", "hit_rate", "success_rate")) + " |")
    md += ["", f"Cleanup: uv run python scripts/purge_cache.py --contains \"LT-{run_id}\""]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
