"""Multi-judge panel orchestration.

Runs N judges on M items sequentially (one model loaded at a time for Ollama
memory management), with resume support and structured JSONL output.
"""

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from judges.judge import Judge, JudgeConfig, JudgeCallResult
from judges.rubrics.base import Rubric

logger = logging.getLogger(__name__)


def load_items(path: str) -> list[dict]:
    """Load evaluation items from a JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _output_path(output_dir: str, task_name: str, model_safe: str) -> str:
    return os.path.join(output_dir, f"{task_name}_{model_safe}.jsonl")


def _load_completed_ids(path: str) -> set[str]:
    """Load item IDs already evaluated from an existing output file."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    item_id = record.get("item_id", "")
                    if item_id:
                        done.add(item_id)
                except json.JSONDecodeError:
                    pass
    return done


def run_judge_on_task(
    judge_config: JudgeConfig,
    rubric: Rubric,
    items: list[dict],
    output_dir: str,
    item_id_key: str = "item_id",
) -> dict:
    """Run a single judge on all items for a task, with resume.

    Returns a summary dict with counts and timing.
    """
    os.makedirs(output_dir, exist_ok=True)
    judge = Judge(judge_config)
    out_path = _output_path(output_dir, rubric.task_name, judge.model_safe)

    done_ids = _load_completed_ids(out_path)
    remaining = [item for item in items if item.get(item_id_key, "") not in done_ids]

    logger.info(
        "Judge %s on %s: %d done, %d remaining out of %d total",
        judge_config.model, rubric.task_name,
        len(done_ids), len(remaining), len(items),
    )

    if not remaining:
        return {
            "judge": judge_config.model,
            "task": rubric.task_name,
            "total": len(items),
            "skipped": len(items),
            "evaluated": 0,
            "errors": 0,
        }

    errors = 0
    total_tokens_in = 0
    total_tokens_out = 0
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(remaining):
            item_id = item.get(item_id_key, f"item_{i}")
            user_prompt = rubric.render_prompt(item)

            result: JudgeCallResult = judge.evaluate(
                system_prompt=rubric.system_prompt,
                user_prompt=user_prompt,
                json_schema=rubric.json_schema,
                expected_fields=rubric.score_fields,
                score_range=rubric.score_range,
            )

            record = {
                "item_id": item_id,
                "judge_model": judge_config.model,
                "task": rubric.task_name,
                "scores": result.scores,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(result.latency_ms, 1),
                "error": result.error,
                "timestamp": datetime.utcnow().isoformat(),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            total_tokens_in += result.input_tokens
            total_tokens_out += result.output_tokens
            if result.error:
                errors += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
                elapsed = time.time() - t0
                logger.info(
                    "  [%d/%d] %s — %.0fs elapsed, %d errors",
                    i + 1, len(remaining), judge_config.model,
                    elapsed, errors,
                )

    elapsed = time.time() - t0
    summary = {
        "judge": judge_config.model,
        "task": rubric.task_name,
        "total": len(items),
        "skipped": len(done_ids),
        "evaluated": len(remaining),
        "errors": errors,
        "total_input_tokens": total_tokens_in,
        "total_output_tokens": total_tokens_out,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("Judge %s on %s complete: %s", judge_config.model, rubric.task_name, summary)
    return summary


def run_panel(
    judge_configs: list[JudgeConfig],
    rubric: Rubric,
    items: list[dict],
    output_dir: str,
    item_id_key: str = "item_id",
) -> list[dict]:
    """Run all judges in the panel sequentially on a task.

    Sequential execution ensures only one large model is loaded in Ollama
    at a time, avoiding memory pressure on the M3 Ultra.
    """
    summaries = []
    for jc in judge_configs:
        summary = run_judge_on_task(jc, rubric, items, output_dir, item_id_key)
        summaries.append(summary)
    return summaries
