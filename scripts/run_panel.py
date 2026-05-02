#!/usr/bin/env python3
"""Run the judge panel (or a single judge) on a task.

Usage:
    python scripts/run_panel.py --task tf1_generation
    python scripts/run_panel.py --task tf2_translation --group proprietary_baselines
    python scripts/run_panel.py --task tf3_generation --group panel --output-dir artifacts/run1
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.config import load_judge_configs, load_task_config, get_rubric, available_tasks
from judges.panel import load_items, run_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run judge panel on a task")
    parser.add_argument("--task", required=True, choices=available_tasks(),
                        help="Task to evaluate")
    parser.add_argument("--group", default="panel",
                        choices=["panel", "proprietary_baselines", "specialized_baseline"],
                        help="Judge group to run (default: panel)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: artifacts/<timestamp>)")
    args = parser.parse_args()

    task_cfg = load_task_config(args.task)
    rubric = get_rubric(args.task)
    judge_configs = load_judge_configs(args.group)

    if not judge_configs:
        logger.error("No judges configured for group '%s'", args.group)
        sys.exit(1)

    data_path = task_cfg["data_path"]
    if not Path(data_path).exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    items = load_items(data_path)
    logger.info("Loaded %d items from %s", len(items), data_path)

    output_dir = args.output_dir or f"artifacts/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    summaries = run_panel(
        judge_configs=judge_configs,
        rubric=rubric,
        items=items,
        output_dir=output_dir,
        item_id_key=task_cfg.get("item_id_key", "item_id"),
    )

    summary_path = Path(output_dir) / f"{args.task}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
