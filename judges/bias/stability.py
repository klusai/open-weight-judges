"""Repeated-run stability audit: measure intra-rater reliability."""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from judges.judge import Judge, JudgeConfig
from judges.panel import load_items
from judges.rubrics.base import Rubric

logger = logging.getLogger(__name__)


def run_stability_audit(
    task: str,
    results_dir: str,
    output_dir: str | None = None,
    n_items: int = 100,
    seed: int = 42,
):
    """Re-run panel on a subset with identical settings, measure per-judge
    score consistency and inter-run agreement.
    """
    from judges.config import load_judge_configs, load_task_config, get_rubric

    task_cfg = load_task_config(task)
    rubric = get_rubric(task)
    dimensions = task_cfg["dimensions"]

    original_by_judge: dict[str, dict[str, dict]] = {}
    for f in sorted(Path(results_dir).glob(f"{task}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge = items[0].get("judge_model", f.stem)
            original_by_judge[judge] = {item["item_id"]: item for item in items}

    all_ids = set()
    for jitems in original_by_judge.values():
        all_ids.update(jitems.keys())

    random.seed(seed)
    selected_ids = sorted(random.sample(sorted(all_ids), min(n_items, len(all_ids))))

    sample_items = load_items(task_cfg["data_path"])
    sample_map = {item["item_id"]: item for item in sample_items}

    judge_configs = load_judge_configs("panel")

    out_dir = output_dir or f"artifacts/bias_audit/{task}_stability"
    os.makedirs(out_dir, exist_ok=True)

    for jc in judge_configs:
        judge = Judge(jc)
        safe = judge.model_safe
        out_path = os.path.join(out_dir, f"{task}_stability_{safe}.jsonl")

        logger.info("Stability audit: %s on %d items", jc.model, len(selected_ids))

        with open(out_path, "w") as out_f:
            for item_id in selected_ids:
                item = sample_map.get(item_id)
                if not item:
                    continue

                user_prompt = rubric.render_prompt(item)
                result = judge.evaluate(
                    system_prompt=rubric.system_prompt,
                    user_prompt=user_prompt,
                    json_schema=rubric.json_schema,
                    expected_fields=rubric.score_fields,
                    score_range=rubric.score_range,
                )

                record = {
                    "item_id": item_id,
                    "judge_model": jc.model,
                    "rerun_scores": result.scores,
                    "error": result.error,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _analyze_stability(original_by_judge, out_dir, task, dimensions, selected_ids)


def _analyze_stability(original_by_judge, audit_dir, task, dimensions, selected_ids):
    """Compare original vs rerun scores."""
    rows = []
    for f in sorted(Path(audit_dir).glob(f"{task}_stability_*.jsonl")):
        rerun_items = load_items(str(f))
        if not rerun_items:
            continue
        judge = rerun_items[0].get("judge_model", f.stem)
        originals = original_by_judge.get(judge, {})

        for rr in rerun_items:
            item_id = rr["item_id"]
            orig = originals.get(item_id, {})
            orig_scores = orig.get("scores", {})
            rr_scores = rr.get("rerun_scores", {})

            for dim in dimensions:
                o = orig_scores.get(dim)
                r = rr_scores.get(dim)
                if isinstance(o, dict):
                    o = o.get("score")
                if isinstance(r, dict):
                    r = r.get("score")
                if o is not None and r is not None:
                    rows.append({
                        "judge": judge,
                        "item_id": item_id,
                        "dimension": dim,
                        "original": o,
                        "rerun": r,
                        "delta": abs(r - o),
                        "exact_match": int(o == r),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No stability audit data to analyze")
        return

    summary = df.groupby(["judge", "dimension"]).agg(
        exact_match_rate=("exact_match", "mean"),
        mean_abs_delta=("delta", "mean"),
        n=("delta", "count"),
    ).reset_index()

    summary_path = os.path.join(audit_dir, f"{task}_stability_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Stability audit summary saved to %s", summary_path)
    print(summary.to_string(index=False))
