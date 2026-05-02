#!/usr/bin/env python3
"""Run bias audit suite on judge panel outputs.

Usage:
    python scripts/run_bias_audit.py --task tf1_generation --results-dir artifacts/run1 --audit position
    python scripts/run_bias_audit.py --task tf1_generation --results-dir artifacts/run1 --audit all
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.config import available_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDITS = ["position", "length", "family", "stability", "all"]


def main():
    parser = argparse.ArgumentParser(description="Run bias audits on judge outputs")
    parser.add_argument("--task", required=True, choices=available_tasks())
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--audit", required=True, choices=AUDITS)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    audits_to_run = AUDITS[:-1] if args.audit == "all" else [args.audit]

    for audit_name in audits_to_run:
        logger.info("Running %s bias audit for task %s", audit_name, args.task)

        if audit_name == "position":
            from judges.bias.position import run_position_audit
            run_position_audit(args.task, args.results_dir, args.output_dir)
        elif audit_name == "length":
            from judges.bias.length import run_length_audit
            run_length_audit(args.task, args.results_dir, args.output_dir)
        elif audit_name == "family":
            from judges.bias.family import run_family_audit
            run_family_audit(args.task, args.results_dir, args.output_dir)
        elif audit_name == "stability":
            from judges.bias.stability import run_stability_audit
            run_stability_audit(args.task, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
