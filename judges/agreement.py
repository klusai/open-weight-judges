"""Inter-judge agreement metrics.

Supports:
- Item-level: ordinal Krippendorff's alpha, weighted Cohen's kappa, Gwet's AC2
- System-level: Kendall tau, Spearman rho
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def krippendorff_alpha(
    scores_by_judge: dict[str, list[dict]],
    dimension: str,
    level: str = "ordinal",
) -> float:
    """Compute Krippendorff's alpha for a single dimension across judges.

    Args:
        scores_by_judge: {judge_model: [{item_id, scores: {dim: val}}, ...]}
        dimension: The score dimension to compute alpha for.
        level: Measurement level ("ordinal", "interval", "nominal", "ratio").
    """
    try:
        import krippendorff as ka
    except ImportError:
        logger.error("krippendorff package required: pip install krippendorff")
        return float("nan")

    item_scores = _build_rating_matrix(scores_by_judge, dimension)
    if not item_scores:
        return float("nan")

    all_items = sorted(item_scores.keys())
    judges = sorted(scores_by_judge.keys())

    matrix = []
    for judge in judges:
        row = []
        for item_id in all_items:
            val = item_scores[item_id].get(judge)
            row.append(val if val is not None else np.nan)
        matrix.append(row)

    reliability_data = np.array(matrix, dtype=float)
    return ka.alpha(reliability_data=reliability_data, level_of_measurement=level)


def pairwise_weighted_kappa(
    scores_judge_a: list[dict],
    scores_judge_b: list[dict],
    dimension: str,
) -> float:
    """Compute linearly weighted Cohen's kappa between two judges."""
    a_map = {item["item_id"]: _extract(item["scores"], dimension) for item in scores_judge_a}
    b_map = {item["item_id"]: _extract(item["scores"], dimension) for item in scores_judge_b}

    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    a_vals = [a_map[k] for k in common if a_map[k] is not None and b_map[k] is not None]
    b_vals = [b_map[k] for k in common if a_map[k] is not None and b_map[k] is not None]

    if len(a_vals) < 2:
        return float("nan")

    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a_vals, b_vals, weights="linear")


def spearman_correlation(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
) -> tuple[float, float]:
    """Compute Spearman rank correlation between two score dictionaries.

    Returns (rho, p_value).
    """
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if len(common) < 3:
        return float("nan"), float("nan")

    a = [scores_a[k] for k in common]
    b = [scores_b[k] for k in common]
    rho, p = stats.spearmanr(a, b)
    return float(rho), float(p)


def kendall_tau(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
) -> tuple[float, float]:
    """Compute Kendall's tau rank correlation.

    Returns (tau, p_value).
    """
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if len(common) < 3:
        return float("nan"), float("nan")

    a = [scores_a[k] for k in common]
    b = [scores_b[k] for k in common]
    tau, p = stats.kendalltau(a, b)
    return float(tau), float(p)


def agreement_summary(
    scores_by_judge: dict[str, list[dict]],
    dimensions: list[str],
) -> pd.DataFrame:
    """Compute agreement metrics for all dimensions.

    Returns a DataFrame with columns [dimension, krippendorff_alpha, mean_kappa].
    """
    judges = sorted(scores_by_judge.keys())
    rows = []

    for dim in dimensions:
        alpha = krippendorff_alpha(scores_by_judge, dim)

        kappas = []
        for i, j1 in enumerate(judges):
            for j2 in judges[i + 1:]:
                k = pairwise_weighted_kappa(
                    scores_by_judge[j1], scores_by_judge[j2], dim
                )
                if not np.isnan(k):
                    kappas.append(k)

        rows.append({
            "dimension": dim,
            "krippendorff_alpha": round(alpha, 4),
            "mean_weighted_kappa": round(np.mean(kappas), 4) if kappas else None,
            "min_weighted_kappa": round(min(kappas), 4) if kappas else None,
            "n_judge_pairs": len(kappas),
        })

    return pd.DataFrame(rows)


def _build_rating_matrix(
    scores_by_judge: dict[str, list[dict]],
    dimension: str,
) -> dict[str, dict[str, float]]:
    """Build {item_id: {judge: score}} mapping."""
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for judge, items in scores_by_judge.items():
        for item in items:
            val = _extract(item.get("scores", {}), dimension)
            if val is not None:
                result[item["item_id"]][judge] = val
    return dict(result)


def _extract(scores: dict, dim: str) -> float | None:
    """Extract numeric score, handling nested {score: N} format."""
    val = scores.get(dim)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None
