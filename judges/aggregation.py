"""Score aggregation methods for multi-judge panels.

Supports:
- Across-judges: median (primary), mean (sensitivity)
- System-level: win-rate, Bradley-Terry ranking
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def median_aggregate(scores_by_judge: dict[str, list[dict]], dimensions: list[str]) -> pd.DataFrame:
    """Compute per-item median score across judges.

    Args:
        scores_by_judge: {judge_model: [{item_id, scores: {dim: val}}, ...]}
        dimensions: List of score dimension names.

    Returns:
        DataFrame with columns [item_id, dim1_median, dim2_median, ...].
    """
    item_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for judge, items in scores_by_judge.items():
        for item in items:
            item_id = item["item_id"]
            scores = item.get("scores", {})
            for dim in dimensions:
                val = _extract_score(scores, dim)
                if val is not None:
                    item_scores[item_id][dim].append(val)

    rows = []
    for item_id, dim_vals in item_scores.items():
        row = {"item_id": item_id}
        for dim in dimensions:
            vals = dim_vals.get(dim, [])
            row[f"{dim}_median"] = float(np.median(vals)) if vals else None
            row[f"{dim}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{dim}_std"] = float(np.std(vals)) if vals else None
            row[f"{dim}_n"] = len(vals)
        rows.append(row)

    return pd.DataFrame(rows)


def system_level_means(
    scores_by_judge: dict[str, list[dict]],
    dimensions: list[str],
    system_key: str = "system_id",
) -> pd.DataFrame:
    """Compute system-level mean scores (e.g., per generator model).

    Groups items by system_key, then averages the panel median per system.
    """
    item_scores: dict[str, dict] = {}

    for judge, items in scores_by_judge.items():
        for item in items:
            item_id = item["item_id"]
            if item_id not in item_scores:
                item_scores[item_id] = {
                    "system": item.get(system_key, "unknown"),
                    "dims": defaultdict(list),
                }
            scores = item.get("scores", {})
            for dim in dimensions:
                val = _extract_score(scores, dim)
                if val is not None:
                    item_scores[item_id]["dims"][dim].append(val)

    rows = []
    for item_id, data in item_scores.items():
        row = {"item_id": item_id, "system": data["system"]}
        for dim in dimensions:
            vals = data["dims"].get(dim, [])
            row[dim] = float(np.median(vals)) if vals else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if "system" in df.columns:
        return df.groupby("system")[dimensions].mean().reset_index()
    return df


def win_rate(
    panel_ranks: dict[str, float],
    baseline_ranks: dict[str, float],
) -> float:
    """Fraction of systems where panel ranking matches baseline ranking direction."""
    systems = set(panel_ranks.keys()) & set(baseline_ranks.keys())
    if len(systems) < 2:
        return float("nan")

    concordant = 0
    total = 0
    systems_list = sorted(systems)
    for i, s1 in enumerate(systems_list):
        for s2 in systems_list[i + 1:]:
            p_diff = panel_ranks[s1] - panel_ranks[s2]
            b_diff = baseline_ranks[s1] - baseline_ranks[s2]
            if p_diff * b_diff > 0:
                concordant += 1
            total += 1

    return concordant / total if total > 0 else float("nan")


def _extract_score(scores: dict, dim: str) -> float | None:
    """Extract a numeric score, handling nested {score: N} format."""
    val = scores.get(dim)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None
