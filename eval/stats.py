"""Paired significance tests for eval compare runs. No scipy dependency."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _normal_sf(z: float) -> float:
    """One-sided survival function 1 - Phi(z) via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank on paired differences (zero diffs dropped)."""
    nonzero = [float(value) for value in diffs if value is not None and float(value) != 0.0]
    n = len(nonzero)
    if n < 10:
        return {"n": n, "statistic": None, "p_value": None, "note": "n<10"}
    ordered = sorted(range(n), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[ordered[j + 1]]) == abs(nonzero[ordered[i]]):
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[ordered[k]] = avg
        i = j + 1
    w_pos = sum(ranks[index] for index in range(n) if nonzero[index] > 0)
    w_neg = sum(ranks[index] for index in range(n) if nonzero[index] < 0)
    statistic = min(w_pos, w_neg)
    mean = n * (n + 1) / 4.0
    # Tie correction for variance.
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # Continuity correction.
    z = (statistic - mean - 0.5) / math.sqrt(var) if var > 0 else 0.0
    p = min(1.0, 2.0 * _normal_sf(abs(z)))
    return {
        "n": n,
        "statistic": round(statistic, 4),
        "p_value": round(p, 6),
        "w_pos": round(w_pos, 4),
        "w_neg": round(w_neg, 4),
    }


def mcnemar(a_true_b_false: int, a_false_b_true: int) -> dict:
    """Two-sided McNemar on discordant binary outcomes (with continuity correction)."""
    b = int(a_true_b_false)
    c = int(a_false_b_true)
    n = b + c
    if n == 0:
        return {"n_discordant": 0, "statistic": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Chi-square(1) survival via normal of sqrt.
    p = min(1.0, 2.0 * _normal_sf(math.sqrt(chi2)))
    return {
        "n_discordant": n,
        "a_hit_b_miss": b,
        "a_miss_b_hit": c,
        "statistic": round(chi2, 4),
        "p_value": round(p, 6),
    }


def _load_items(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    items = path / "items.jsonl"
    if not items.is_file():
        raise FileNotFoundError(items)
    for line in items.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row.get("id"))] = row
    return rows


def _item_mrr(row: dict) -> float | None:
    if not row.get("answerable"):
        return None
    stages = row.get("stages") or {}
    value = (stages.get("reranked") or {}).get("mrr")
    if value is None:
        value = row.get("mrr")
    if value is None:
        return None
    return float(value)


def _item_hit(row: dict) -> bool | None:
    if not row.get("answerable"):
        return None
    stages = row.get("stages") or {}
    value = (stages.get("reranked") or {}).get("hit")
    if value is None:
        value = row.get("hit_at_8")
    if value is None:
        return None
    return bool(value)


def paired_against(current_dir: Path, baseline_dir: Path, label: str) -> dict:
    current = _load_items(Path(current_dir))
    baseline = _load_items(Path(baseline_dir))
    shared = sorted(set(current) & set(baseline))
    mrr_diffs = []
    a_hit_b_miss = 0
    a_miss_b_hit = 0
    for item_id in shared:
        left = current[item_id]
        right = baseline[item_id]
        left_mrr = _item_mrr(left)
        right_mrr = _item_mrr(right)
        if left_mrr is not None and right_mrr is not None:
            mrr_diffs.append(left_mrr - right_mrr)
        left_hit = _item_hit(left)
        right_hit = _item_hit(right)
        if left_hit is None or right_hit is None:
            continue
        if left_hit and not right_hit:
            a_hit_b_miss += 1
        elif (not left_hit) and right_hit:
            a_miss_b_hit += 1
    return {
        "baseline": label,
        "baseline_dir": str(baseline_dir),
        "n_shared": len(shared),
        "wilcoxon_mrr": wilcoxon_signed_rank(mrr_diffs),
        "mcnemar_hit": mcnemar(a_hit_b_miss, a_miss_b_hit),
    }
