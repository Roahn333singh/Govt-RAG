"""Production latency + cited-source MRR test (50 miss-pass, 40 hit-pass = 90 /chat calls).

Usage:
  export PROD_API_KEY=...            # never printed
  uv run python eval/prod50_test.py [--dry-run] [--url URL] [--gap 4.5] [--hit-n 40]
"""
from __future__ import annotations

import argparse, json, os, re, statistics, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import normalize_text, sources_match, split_parts  # noqa: E402

DEFAULT_URL = "https://manual-ai-rag-nszha5soya-ey.a.run.app"
REFUSAL_MARKERS = ("I am an AI assistant specialized", "does not contain any information",
                   "does not contain information about", "no information about", "does not specify")


def gold_page(item):
    m = re.search(r"\d+", str(item.get("source_page_or_section") or ""))
    return int(m.group()) if m else None


def score(item, sources):
    """Return (rr, method, position). Positions are 1-based in the cited source list."""
    docs = split_parts(item.get("source_doc"))
    parts = [normalize_text(p) for p in split_parts(item.get("evidence_quote"))]
    parts = [p for p in parts if p]
    for pos, s in enumerate(sources, 1):
        if not sources_match(s.get("source_document"), docs):
            continue
        hay = normalize_text(s.get("content"))
        if any(p in hay for p in parts):
            return 1.0 / pos, "text", pos
    page = gold_page(item)
    if page is not None:
        for pos, s in enumerate(sources, 1):
            if sources_match(s.get("source_document"), docs) and str(s.get("page_start")) == str(page):
                return 1.0 / pos, "doc_page", pos
    return 0.0, "none", None


def is_refusal(answer, sources):
    return (not sources) or any(m in (answer or "") for m in REFUSAL_MARKERS)


def post_chat(url, key, question, timeout=90):
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, rid, raw = r.status, r.headers.get("X-Request-ID"), r.read()
    except urllib.error.HTTPError as e:
        status, rid, raw = e.code, e.headers.get("X-Request-ID"), e.read()
    except Exception as e:  # timeout / network
        return {"status": None, "error": type(e).__name__, "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        data = {}
    return {"status": status, "wall_s": wall, "request_id": rid, "cached": data.get("cached"),
            "answer": data.get("answer"), "sources": data.get("sources") or [],
            "detail": data.get("detail")}


def stats(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    pct = lambda q: s[min(len(s) - 1, max(0, int(round(q * len(s) + 0.5)) - 1))]
    return {"n": len(s), "p50": round(statistics.median(s), 3), "p95": round(pct(0.95), 3),
            "max": round(s[-1], 3), "mean": round(statistics.mean(s), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--url", default=os.getenv("PROD_URL") or DEFAULT_URL)
    ap.add_argument("--items", default=str(ROOT / "eval" / "prod50.jsonl"))
    ap.add_argument("--gap", type=float, default=4.5, help="seconds between request starts")
    ap.add_argument("--hit-n", type=int, default=40)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    plan = [("miss", it) for it in items] + [("hit", it) for it in items[: a.hit_n]]
    print(f"URL {a.url} | {len(items)} items | {len(plan)} /chat calls | gap {a.gap}s "
          f"| est. >= {len(plan) * a.gap / 60:.1f} min")
    if a.dry_run:
        print("GET /health (warm-up)")
        for i, (p, it) in enumerate(plan, 1):
            print(f"{i:3d} {p:4s} POST /chat {it['id']} [{it['language']}] {it['question'][:70]}")
        print("dry run: nothing sent")
        return
    key = os.getenv("PROD_API_KEY")
    if not key:
        sys.exit("PROD_API_KEY is not set")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "eval" / "results" / f"prod50_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=90) as r:
            print("warm-up /health", r.status, r.read()[:120].decode(errors="replace"))
    except Exception as e:
        print("warm-up failed:", type(e).__name__)

    rows, n429 = [], 0
    with open(out / "raw.jsonl", "w", encoding="utf-8") as raw:
        for i, (p, it) in enumerate(plan, 1):
            start = time.monotonic()
            res = post_chat(a.url, key, it["question"])
            if res.get("status") == 429:
                n429 += 1
                print("  429 -> waiting 60s, retry once")
                time.sleep(60)
                res = post_chat(a.url, key, it["question"])
                res["retried"] = True
                if res.get("status") == 429:
                    n429 += 1
            ok = res.get("status") == 200
            refusal = ok and is_refusal(res.get("answer"), res["sources"])
            rr, method, pos = score(it, res["sources"]) if ok and not refusal else (0.0, "none", None)
            row = {"n": i, "pass": p, "id": it["id"], "lang": it["language"], "type": it["type"],
                   "question": it["question"], **res, "refusal": refusal, "rr": rr,
                   "match": method, "position": pos}
            rows.append(row)
            raw.write(json.dumps(row, ensure_ascii=False) + "\n"); raw.flush()
            print(f"{i:3d}/{len(plan)} {p:4s} {it['id']} {it['language']} status={res.get('status')} "
                  f"cached={res.get('cached')} {res['wall_s']:.2f}s rr={rr:.2f} {method}"
                  f"{' REFUSAL' if refusal else ''} rid={res.get('request_id')}")
            wait = a.gap - (time.monotonic() - start)
            if wait > 0 and i < len(plan):
                time.sleep(wait)

    ok = [r for r in rows if r.get("status") == 200]
    p1 = [r for r in ok if r["pass"] == "miss"]
    def mrr(rs):
        return round(sum(r["rr"] for r in rs) / len(rs), 4) if rs else None
    summary = {
        "url": a.url, "utc": stamp, "requests": len(rows), "ok": len(ok),
        "errors": sum(1 for r in rows if r.get("status") != 200),
        "http_429": n429,
        "refusals_pass1": sum(r["refusal"] for r in p1),
        "mrr_cited": {"overall": mrr(p1), "en": mrr([r for r in p1 if r["lang"] == "en"]),
                      "hi": mrr([r for r in p1 if r["lang"] == "hi"])},
        "cited_hit_rate": round(sum(r["rr"] > 0 for r in p1) / len(p1), 4) if p1 else None,
        "match_method_counts": {m: sum(r["match"] == m for r in p1) for m in ("text", "doc_page", "none")},
        "latency_s": {
            "cached_false": stats([r["wall_s"] for r in ok if r.get("cached") is False]),
            "cached_true": stats([r["wall_s"] for r in ok if r.get("cached") is True]),
            "pass1": stats([r["wall_s"] for r in p1]),
            "pass2": stats([r["wall_s"] for r in ok if r["pass"] == "hit"]),
        },
        "cached_in_pass1": [r["id"] for r in p1 if r.get("cached")],
        "missed_in_pass2": [r["id"] for r in ok if r["pass"] == "hit" and not r.get("cached")],
        "low_or_missing_rank": [r["id"] for r in p1 if r["rr"] < 0.5],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    L = summary["latency_s"]
    md = [f"# prod50 test {stamp}", "", f"URL: {a.url}", "",
          "## Retrieval (pass 1, cited sources)", "",
          f"- MRR overall {summary['mrr_cited']['overall']} | en {summary['mrr_cited']['en']} | hi {summary['mrr_cited']['hi']}",
          f"- cited hit rate {summary['cited_hit_rate']}",
          f"- match method {summary['match_method_counts']}", "",
          "## Latency (client wall time, s)", "",
          "| group | n | p50 | p95 | max | mean |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in L.items():
        md.append(f"| {k} | {v.get('n')} | {v.get('p50', '—')} | {v.get('p95', '—')} | {v.get('max', '—')} | {v.get('mean', '—')} |")
    md += ["", "## Counts", "",
           f"- requests {summary['requests']}, ok {summary['ok']}, errors {summary['errors']}, 429s {n429}",
           f"- refusals in pass 1: {summary['refusals_pass1']}",
           f"- cached already in pass 1: {summary['cached_in_pass1']}",
           f"- missed again in pass 2: {summary['missed_in_pass2']}",
           f"- rr < 0.5 in pass 1: {summary['low_or_missing_rank']}"]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md[4:]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
