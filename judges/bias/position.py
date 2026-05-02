"""Position bias audit: swap prompt/fable order and measure score sensitivity."""

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


def _swap_prompt_fable(rubric: Rubric, item: dict) -> str:
    """Re-render prompt with fable and prompt fields swapped in order."""
    template = rubric.evaluation_prompt_template
    fable_key = rubric.text_field_map.get("fable", "fable")
    prompt_key = rubric.text_field_map.get("prompt", "prompt")

    fable_text = item.get(fable_key, "")
    prompt_text = item.get(prompt_key, "")

    swapped = template
    fable_placeholder = "{{fable}}"
    prompt_placeholder = "{{prompt}}"

    if fable_placeholder in swapped and prompt_placeholder in swapped:
        fi = swapped.index(fable_placeholder)
        pi = swapped.index(prompt_placeholder)
        if fi > pi:
            swapped = swapped.replace(prompt_placeholder, "___FABLE___", 1)
            swapped = swapped.replace(fable_placeholder, "___PROMPT___", 1)
        else:
            swapped = swapped.replace(fable_placeholder, "___PROMPT___", 1)
            swapped = swapped.replace(prompt_placeholder, "___FABLE___", 1)
        swapped = swapped.replace("___FABLE___", fable_text)
        swapped = swapped.replace("___PROMPT___", prompt_text)
    else:
        swapped = rubric.render_prompt(item)

    return swapped


def run_position_audit(
    task: str,
    results_dir: str,
    output_dir: str | None = None,
    n_items: int = 200,
    seed: int = 42,
):
    """Swap prompt/fable order, re-score a subset, compare with original scores.

    Args:
        task: Task name (e.g., "tf1_generation").
        results_dir: Directory with original panel JSONL outputs.
        output_dir: Where to save audit results.
        n_items: Number of items to re-score with swapped order.
        seed: Random seed for item selection.
    """
    from judges.config import load_judge_configs, load_task_config, get_rubric

    task_cfg = load_task_config(task)
    rubric = get_rubric(task)
    dimensions = task_cfg["dimensions"]

    original_by_judge: dict[str, dict[str, dict]] = {}
    results_path = Path(results_dir)
    for f in sorted(results_path.glob(f"{task}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge = items[0].get("judge_model", f.stem)
            original_by_judge[judge] = {item["item_id"]: item for item in items}

    if not original_by_judge:
        logger.error("No panel results found in %s for task %s", results_dir, task)
        return

    all_item_ids = set()
    for judge_items in original_by_judge.values():
        all_item_ids.update(judge_items.keys())

    random.seed(seed)
    selected_ids = sorted(random.sample(sorted(all_item_ids), min(n_items, len(all_item_ids))))

    sample_items = load_items(task_cfg["data_path"])
    sample_map = {item["item_id"]: item for item in sample_items}

    judge_configs = load_judge_configs("panel")

    out_dir = output_dir or f"artifacts/bias_audit/{task}_position"
    os.makedirs(out_dir, exist_ok=True)

    for jc in judge_configs:
        judge = Judge(jc)
        safe = judge.model_safe
        out_path = os.path.join(out_dir, f"{task}_position_{safe}.jsonl")

        logger.info("Position audit: %s on %d items", jc.model, len(selected_ids))

        with open(out_path, "w") as out_f:
            for item_id in selected_ids:
                item = sample_map.get(item_id)
                if not item:
                    continue

                swapped_prompt = _swap_prompt_fable(rubric, item)
                result = judge.evaluate(
                    system_prompt=rubric.system_prompt,
                    user_prompt=swapped_prompt,
                    json_schema=rubric.json_schema,
                    expected_fields=rubric.score_fields,
                    score_range=rubric.score_range,
                )

                record = {
                    "item_id": item_id,
                    "judge_model": jc.model,
                    "swapped_scores": result.scores,
                    "error": result.error,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _analyze_position_results(original_by_judge, out_dir, task, dimensions, selected_ids)


def _analyze_position_results(
    original_by_judge, audit_dir, task, dimensions, selected_ids
):
    """Compare original vs swapped scores."""
    rows = []
    for f in sorted(Path(audit_dir).glob(f"{task}_position_*.jsonl")):
        swapped_items = load_items(str(f))
        if not swapped_items:
            continue
        judge = swapped_items[0].get("judge_model", f.stem)
        originals = original_by_judge.get(judge, {})

        for sw in swapped_items:
            item_id = sw["item_id"]
            orig = originals.get(item_id, {})
            orig_scores = orig.get("scores", {})
            sw_scores = sw.get("swapped_scores", {})

            for dim in dimensions:
                o = orig_scores.get(dim)
                s = sw_scores.get(dim)
                if isinstance(o, dict):
                    o = o.get("score")
                if isinstance(s, dict):
                    s = s.get("score")
                if o is not None and s is not None:
                    rows.append({
                        "judge": judge,
                        "item_id": item_id,
                        "dimension": dim,
                        "original": o,
                        "swapped": s,
                        "delta": s - o,
                        "exact_match": int(o == s),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No position audit data to analyze")
        return

    summary = df.groupby(["judge", "dimension"]).agg(
        exact_match_rate=("exact_match", "mean"),
        mean_abs_delta=("delta", lambda x: np.mean(np.abs(x))),
        mean_delta=("delta", "mean"),
        n=("delta", "count"),
    ).reset_index()

    summary_path = os.path.join(audit_dir, f"{task}_position_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Position audit summary saved to %s", summary_path)
    print(summary.to_string(index=False))
