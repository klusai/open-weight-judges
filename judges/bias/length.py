"""Length bias audit: analyze whether text length predicts scores."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from judges.panel import load_items

logger = logging.getLogger(__name__)


def run_length_audit(task: str, results_dir: str, output_dir: str | None = None):
    """Bin items by text length, compare score distributions, run regression.

    Tests whether length predicts score after controlling for quality.
    """
    from judges.config import load_task_config, get_rubric

    task_cfg = load_task_config(task)
    rubric = get_rubric(task)
    dimensions = task_cfg["dimensions"]

    sample_items = load_items(task_cfg["data_path"])
    length_map = {}
    for item in sample_items:
        fable = item.get("fable", "")
        length_map[item["item_id"]] = len(fable.split())

    scores_by_judge: dict[str, list[dict]] = {}
    for f in sorted(Path(results_dir).glob(f"{task}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge = items[0].get("judge_model", f.stem)
            scores_by_judge[judge] = items

    rows = []
    for judge, items in scores_by_judge.items():
        for item in items:
            item_id = item["item_id"]
            word_count = length_map.get(item_id, 0)
            scores = item.get("scores", {})
            for dim in dimensions:
                val = scores.get(dim)
                if isinstance(val, dict):
                    val = val.get("score")
                if val is not None:
                    rows.append({
                        "judge": judge,
                        "item_id": item_id,
                        "dimension": dim,
                        "score": val,
                        "word_count": word_count,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No data for length bias analysis")
        return

    terciles = df["word_count"].quantile([0.33, 0.67])
    df["length_bin"] = pd.cut(
        df["word_count"],
        bins=[0, terciles.iloc[0], terciles.iloc[1], float("inf")],
        labels=["short", "medium", "long"],
    )

    out_dir = output_dir or f"artifacts/bias_audit/{task}_length"
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    for judge in df["judge"].unique():
        for dim in dimensions:
            subset = df[(df["judge"] == judge) & (df["dimension"] == dim)]
            if len(subset) < 10:
                continue

            for bin_label in ["short", "medium", "long"]:
                bin_data = subset[subset["length_bin"] == bin_label]
                if not bin_data.empty:
                    summary_rows.append({
                        "judge": judge,
                        "dimension": dim,
                        "length_bin": bin_label,
                        "mean_score": bin_data["score"].mean(),
                        "std_score": bin_data["score"].std(),
                        "n": len(bin_data),
                    })

            rho, p = stats.spearmanr(subset["word_count"], subset["score"])
            summary_rows.append({
                "judge": judge,
                "dimension": dim,
                "length_bin": "correlation",
                "mean_score": rho,
                "std_score": p,
                "n": len(subset),
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, f"{task}_length_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Length audit summary saved to %s", summary_path)
    print(summary.to_string(index=False))
