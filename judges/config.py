"""Load YAML configuration and instantiate judge/task objects."""

import os
from pathlib import Path

import yaml

from judges.judge import JudgeConfig
from judges.rubrics import RUBRICS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "conf"


def load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env(val: str) -> str:
    """Replace ${VAR} with environment variable value."""
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        return os.environ.get(val[2:-1], "")
    return val


def load_judge_configs(group: str = "panel") -> list[JudgeConfig]:
    """Load judge configurations from judges.yaml.

    Args:
        group: One of "panel", "proprietary_baselines", "specialized_baseline".
    """
    data = load_yaml("judges.yaml")
    entries = data.get(group, [])
    configs = []
    for entry in entries:
        configs.append(JudgeConfig(
            name=entry["name"],
            model=entry["model"],
            base_url=_resolve_env(entry.get("base_url", "")),
            api_key=_resolve_env(entry.get("api_key", "ollama")),
            temperature=entry.get("temperature", 0.0),
            max_tokens=entry.get("max_tokens", 4096),
            use_strict_schema=entry.get("use_strict_schema", True),
            disable_thinking=entry.get("disable_thinking", False),
            reasoning_effort=entry.get("reasoning_effort"),
        ))
    return configs


def load_task_config(task_name: str) -> dict:
    """Load a single task configuration from tasks.yaml."""
    data = load_yaml("tasks.yaml")
    tasks = data.get("tasks", {})
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(tasks.keys())}")
    return tasks[task_name]


def get_rubric(task_name: str):
    """Get the rubric for a task."""
    task_cfg = load_task_config(task_name)
    rubric_name = task_cfg.get("rubric", task_name)
    if rubric_name not in RUBRICS:
        raise ValueError(f"Unknown rubric: {rubric_name}. Available: {list(RUBRICS.keys())}")
    return RUBRICS[rubric_name]


def available_tasks() -> list[str]:
    """List all configured task names."""
    data = load_yaml("tasks.yaml")
    return list(data.get("tasks", {}).keys())
