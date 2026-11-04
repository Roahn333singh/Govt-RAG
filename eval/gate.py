"""Compare a summary with eval/thresholds.yaml. Pure: no network, no models."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = ROOT / "eval" / "thresholds.yaml"

# Defaults match eval/thresholds.yaml. The file wins when it is present.
_DEFAULTS = {
    "recall_at_8_drop_pts": 3.0,
    "faithfulness_min": 0.90,
    "false_refusal_max": 0.05,
    "correct_refusal_min": 0.90,
    "p95_latency_s_max": 15.0,
    "error_rate_max": 0.02,
}


def load_thresholds(path: Path | None = None) -> dict:
    text = (path or THRESHOLDS).read_text(encoding="utf-8")
    values = dict(_DEFAULTS)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        values[key.strip()] = float(raw.strip())
    return values


def _missing(value) -> bool:
    return value is None


def gate_failures(summary: dict, baseline: dict | None, thresholds: dict | None = None) -> list[str]:
    """Failures past the thresholds. A missing metric (eval not run) is skipped.

    recall@8 uses a drop against the baseline, in percentage points.
    The other checks are absolute.
    """
    limits = thresholds or load_thresholds()
    failures: list[str] = []
    retrieval = summary.get("retrieval") or {}
    answer = summary.get("answer") or {}
    base_retrieval = (baseline or {}).get("retrieval") or {}

    recall = retrieval.get("recall_at_8")
    base_recall = base_retrieval.get("recall_at_8")
    if not _missing(recall) and not _missing(base_recall):
        drop_pts = (float(base_recall) - float(recall)) * 100.0
        allowed = float(limits["recall_at_8_drop_pts"])
        if drop_pts > allowed:
            failures.append(
                f"recall@8 dropped {drop_pts:.1f} pts (limit {allowed:.1f})"
            )

    faithfulness = answer.get("faithfulness")
    if not _missing(faithfulness) and float(faithfulness) < float(limits["faithfulness_min"]):
        failures.append(
            f"faithfulness {float(faithfulness):.3f} < {float(limits['faithfulness_min']):.2f}"
        )

    false_refusal = answer.get("false_refusal_rate")
    if not _missing(false_refusal) and float(false_refusal) > float(limits["false_refusal_max"]):
        failures.append(
            f"false refusal {float(false_refusal):.1%} > {float(limits['false_refusal_max']):.0%}"
        )

    correct_refusal = answer.get("correct_refusal_rate")
    if not _missing(correct_refusal) and float(correct_refusal) < float(limits["correct_refusal_min"]):
        failures.append(
            f"correct refusal {float(correct_refusal):.1%} < {float(limits['correct_refusal_min']):.0%}"
        )

    p95 = answer.get("latency_p95_s")
    if not _missing(p95) and float(p95) > float(limits["p95_latency_s_max"]):
        failures.append(
            f"p95 latency {float(p95):.1f}s > {float(limits['p95_latency_s_max']):.0f}s"
        )

    error_rate = answer.get("error_rate")
    if not _missing(error_rate) and float(error_rate) > float(limits["error_rate_max"]):
        failures.append(
            f"error rate {float(error_rate):.1%} > {float(limits['error_rate_max']):.0%}"
        )
    return failures
