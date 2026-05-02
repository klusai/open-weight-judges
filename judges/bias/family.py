"""Family preference audit: test whether judges rate same-family generators higher."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from judges.panel import load_items

logger = logging.getLogger(__name__)

JUDGE_FAMILIES = {
    "granite4.1:30b": "ibm",
    "granite3.3:8b": "ibm",
    "exaone3.5:32b": "lg",
}

GENERATOR_FAMILIES = {
    "CohereForAI/aya-23-8B": "cohere",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "huggingface",
    "Qwen/Qwen2.5-7B-Instruct": "alibaba",
    "allenai/Llama-3.1-Tulu-3-8B": "meta",
    "deepseek-ai/deepseek-llm-7b-chat": "deepseek",
    "meta-llama/Llama-3.1-8B-Instruct": "meta",
    "meta-llama/Llama-3.2-1B-Instruct": "meta",
    "microsoft/Phi-3-mini-4k-instruct ": "microsoft",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "tiiuae/Falcon3-7B-Instruct": "tii",
}


def run_family_audit(task: str, results_dir: str, output_dir: str | None = None):
    """Compare panel scores for generators from families related to panel judges
    vs unrelated families. Connects to preference leakage literature.
    """
    from judges.config import load_task_config

    task_cfg = load_task_config(task)
    dimensions = task_cfg["dimensions"]

    scores_by_judge: dict[str, list[dict]] = {}
    for f in sorted(Path(results_dir).glob(f"{task}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge = items[0].get("judge_model", f.stem)
            scores_by_judge[judge] = items

    rows = []
    for judge_model, items in scores_by_judge.items():
        judge_family = JUDGE_FAMILIES.get(judge_model, "unknown")

        for item in items:
            gen_model = item.get("generator_model", "unknown")
            gen_family = GENERATOR_FAMILIES.get(gen_model, "unknown")
            same_family = judge_family == gen_family
            scores = item.get("scores", {})

            for dim in dimensions:
                val = scores.get(dim)
                if isinstance(val, dict):
                    val = val.get("score")
                if val is not None:
                    rows.append({
                        "judge": judge_model,
                        "judge_family": judge_family,
                        "generator": gen_model,
                        "generator_family": gen_family,
                        "same_family": same_family,
                        "dimension": dim,
                        "score": val,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No data for family preference analysis")
        return

    out_dir = output_dir or f"artifacts/bias_audit/{task}_family"
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    for judge in df["judge"].unique():
        j_df = df[df["judge"] == judge]
        for dim in dimensions:
            d_df = j_df[j_df["dimension"] == dim]
            same = d_df[d_df["same_family"]]
            diff = d_df[~d_df["same_family"]]

            if len(same) > 0 and len(diff) > 0:
                t_stat, p_val = stats.ttest_ind(same["score"], diff["score"])
            else:
                t_stat, p_val = float("nan"), float("nan")

            summary_rows.append({
                "judge": judge,
                "dimension": dim,
                "same_family_mean": same["score"].mean() if len(same) > 0 else None,
                "diff_family_mean": diff["score"].mean() if len(diff) > 0 else None,
                "delta": (same["score"].mean() - diff["score"].mean()) if len(same) > 0 and len(diff) > 0 else None,
                "t_stat": t_stat,
                "p_value": p_val,
                "n_same": len(same),
                "n_diff": len(diff),
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, f"{task}_family_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Family audit summary saved to %s", summary_path)
    print(summary.to_string(index=False))
