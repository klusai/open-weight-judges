#!/usr/bin/env python3
"""Select high-disagreement items for human arbitration.

Computes panel median vs proprietary baseline scores, ranks items by
absolute disagreement, and exports annotation sheets.

Usage:
    python scripts/select_arbitration_items.py \
        --task tf1_generation \
        --panel-dir artifacts/panel_run \
        --baseline-dir artifacts/baseline_run \
        --n 100 \
        --output artifacts/arbitration/tf1
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.config import load_task_config, available_tasks
from judges.panel import load_items
from judges.aggregation import median_aggregate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_judge_results(results_dir: str, task_name: str) -> dict[str, list[dict]]:
    """Load all judge JSONL outputs for a task."""
    scores_by_judge: dict[str, list[dict]] = {}
    for f in sorted(Path(results_dir).glob(f"{task_name}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge = items[0].get("judge_model", f.stem)
            scores_by_judge[judge] = items
    return scores_by_judge


def extract_score(scores: dict, dim: str) -> float | None:
    """Extract numeric score from flat or nested format."""
    val = scores.get(dim)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def compute_disagreement(
    panel_dir: str,
    baseline_dir: str,
    task_name: str,
    dimensions: list[str],
    n: int = 100,
) -> pd.DataFrame:
    """Compute panel-vs-baseline disagreement and select top N items."""
    panel_by_judge = load_judge_results(panel_dir, task_name)
    baseline_by_judge = load_judge_results(baseline_dir, task_name)

    if not panel_by_judge:
        logger.error("No panel results found in %s for task %s", panel_dir, task_name)
        sys.exit(1)
    if not baseline_by_judge:
        logger.error("No baseline results found in %s for task %s", baseline_dir, task_name)
        sys.exit(1)

    panel_agg = median_aggregate(panel_by_judge, dimensions)

    baseline_judge = list(baseline_by_judge.keys())[0]
    baseline_items = baseline_by_judge[baseline_judge]
    baseline_map = {}
    for item in baseline_items:
        item_id = item["item_id"]
        scores = item.get("scores", {})
        baseline_map[item_id] = {dim: extract_score(scores, dim) for dim in dimensions}

    rows = []
    for _, row in panel_agg.iterrows():
        item_id = row["item_id"]
        if item_id not in baseline_map:
            continue

        deltas = []
        for dim in dimensions:
            panel_val = row.get(f"{dim}_median")
            baseline_val = baseline_map[item_id].get(dim)
            if panel_val is not None and baseline_val is not None:
                deltas.append(abs(panel_val - baseline_val))

        if deltas:
            rows.append({
                "item_id": item_id,
                "mean_abs_delta": np.mean(deltas),
                "max_delta": max(deltas),
                **{f"{dim}_panel": row.get(f"{dim}_median") for dim in dimensions},
                **{f"{dim}_baseline": baseline_map[item_id].get(dim) for dim in dimensions},
            })

    df = pd.DataFrame(rows).sort_values("mean_abs_delta", ascending=False)
    return df.head(n)


def export_annotation_sheet(
    selected: pd.DataFrame,
    samples_path: str,
    output_path: str,
    task_name: str,
    dimensions: list[str],
):
    """Export annotation sheet as CSV (for Google Sheets import)."""
    samples = {item["item_id"]: item for item in load_items(samples_path)}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["item_id", "task"]
        if task_name == "tf2_translation":
            header += ["original_fable", "translated_fable"]
        else:
            header += ["fable", "prompt"]
        header += [f"{dim}_score" for dim in dimensions]
        header += ["notes"]
        writer.writerow(header)

        for _, row in selected.iterrows():
            item_id = row["item_id"]
            sample = samples.get(item_id, {})

            data = [item_id, task_name]
            if task_name == "tf2_translation":
                data += [sample.get("fable", ""), sample.get("translated_fable", "")]
            else:
                data += [sample.get("fable", ""), sample.get("prompt", "")]
            data += [""] * len(dimensions)
            data += [""]
            writer.writerow(data)

    logger.info("Exported %d items to %s", len(selected), output_path)


def main():
    parser = argparse.ArgumentParser(description="Select items for human arbitration")
    parser.add_argument("--task", required=True, choices=available_tasks())
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", required=True, help="Output directory for annotation sheets")
    args = parser.parse_args()

    task_cfg = load_task_config(args.task)
    dimensions = task_cfg["dimensions"]

    selected = compute_disagreement(
        args.panel_dir, args.baseline_dir, args.task, dimensions, args.n
    )

    logger.info(
        "Selected %d items with mean_abs_delta range [%.2f, %.2f]",
        len(selected),
        selected["mean_abs_delta"].min() if len(selected) > 0 else 0,
        selected["mean_abs_delta"].max() if len(selected) > 0 else 0,
    )

    disagreement_path = os.path.join(args.output, f"{args.task}_disagreement.csv")
    selected.to_csv(disagreement_path, index=False)
    logger.info("Disagreement analysis saved to %s", disagreement_path)

    samples_path = task_cfg.get("data_path", f"data/{args.task.split('_')[0]}/samples.jsonl")
    sheet_path = os.path.join(args.output, f"{args.task}_annotation_sheet.csv")
    export_annotation_sheet(selected, samples_path, sheet_path, args.task, dimensions)


if __name__ == "__main__":
    main()
