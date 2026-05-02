#!/usr/bin/env python3
"""Compute agreement and correlation statistics from judge output files.

Usage:
    python scripts/analyze_agreement.py --task tf1_generation --results-dir artifacts/run1
    python scripts/analyze_agreement.py --task tf1_generation --results-dir artifacts/run1 --baseline-dir artifacts/baseline
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.config import load_task_config, available_tasks
from judges.panel import load_items
from judges.aggregation import median_aggregate, system_level_means
from judges.agreement import agreement_summary, spearman_correlation, kendall_tau

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_judge_results(results_dir: str, task_name: str) -> dict[str, list[dict]]:
    """Load all judge JSONL outputs for a task from a results directory."""
    scores_by_judge: dict[str, list[dict]] = {}
    results_path = Path(results_dir)

    for f in sorted(results_path.glob(f"{task_name}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge_model = items[0].get("judge_model", f.stem)
            scores_by_judge[judge_model] = items
            logger.info("Loaded %d items from %s (judge: %s)", len(items), f.name, judge_model)

    return scores_by_judge


def main():
    parser = argparse.ArgumentParser(description="Analyze inter-judge agreement")
    parser.add_argument("--task", required=True, choices=available_tasks())
    parser.add_argument("--results-dir", required=True, help="Directory with judge JSONL outputs")
    parser.add_argument("--baseline-dir", default=None, help="Directory with proprietary baseline outputs")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    task_cfg = load_task_config(args.task)
    dimensions = task_cfg["dimensions"]

    scores_by_judge = load_judge_results(args.results_dir, args.task)
    if len(scores_by_judge) < 2:
        logger.error("Need at least 2 judges, found %d", len(scores_by_judge))
        sys.exit(1)

    logger.info("Computing agreement for %d judges on %d dimensions", len(scores_by_judge), len(dimensions))

    agreement_df = agreement_summary(scores_by_judge, dimensions)
    print("\n=== Inter-Judge Agreement ===")
    print(agreement_df.to_string(index=False))

    agg_df = median_aggregate(scores_by_judge, dimensions)
    print(f"\n=== Aggregated scores for {len(agg_df)} items ===")
    for dim in dimensions:
        col = f"{dim}_median"
        if col in agg_df.columns:
            vals = agg_df[col].dropna()
            print(f"  {dim}: median={vals.median():.2f}, mean={vals.mean():.2f}, std={vals.std():.2f}")

    system_key = task_cfg.get("system_key")
    if system_key:
        sys_df = system_level_means(scores_by_judge, dimensions, system_key)
        if not sys_df.empty:
            print(f"\n=== System-Level Means (by {system_key}) ===")
            print(sys_df.to_string(index=False))

    if args.baseline_dir:
        baseline_by_judge = load_judge_results(args.baseline_dir, args.task)
        if baseline_by_judge:
            baseline_name = list(baseline_by_judge.keys())[0]
            baseline_items = baseline_by_judge[baseline_name]
            logger.info("Loaded baseline: %s (%d items)", baseline_name, len(baseline_items))

    results = {
        "task": args.task,
        "n_judges": len(scores_by_judge),
        "judges": list(scores_by_judge.keys()),
        "agreement": agreement_df.to_dict(orient="records"),
    }

    out_path = args.output or str(Path(args.results_dir) / f"{args.task}_agreement.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
