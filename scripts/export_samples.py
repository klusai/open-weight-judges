#!/usr/bin/env python3
"""Export evaluation samples from TF1/TF2/TF3 repos into standardized JSONL.

Normalizes heterogeneous input formats into a common schema:
    {task, item_id, <text_fields>, <metadata>}

Usage:
    python scripts/export_samples.py --tf1-dir ../tinyfabulist-tf1/data/fables
    python scripts/export_samples.py --tf2-dir ../tinyfabulist-tf2/data
    python scripts/export_samples.py --tf3-dir ../tinyfabulist-tf3/tf3/evaluation/artifacts
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %d items to %s", len(items), path)


def export_tf1(fables_dir: str, output_dir: str = "data/tf1"):
    """Export TF1 fables: 10 models x 100 fables each."""
    items = []
    for fpath in sorted(Path(fables_dir).glob("**/*.jsonl")):
        if "eval" in fpath.name or "stat" in fpath.name:
            continue
        for entry in load_jsonl(str(fpath)):
            if "fable" not in entry:
                continue
            item_id = entry.get("hash", hashlib.md5(entry["fable"].encode()).hexdigest()[:12])
            items.append({
                "task": "tf1_generation",
                "item_id": item_id,
                "fable": entry["fable"],
                "prompt": entry.get("prompt", ""),
                "generator_model": entry.get("llm_name", "unknown"),
            })

    save_jsonl(items, os.path.join(output_dir, "samples.jsonl"))
    logger.info("TF1: exported %d fables from %s", len(items), fables_dir)


def export_tf2(data_dir: str, output_dir: str = "data/tf2"):
    """Export TF2 translation pairs from evaluation set."""
    items = []
    for fpath in sorted(Path(data_dir).glob("**/*.jsonl")):
        for entry in load_jsonl(str(fpath)):
            if "translated_fable" not in entry:
                continue
            item_id = entry.get("hash", hashlib.md5(
                entry.get("fable", "").encode()
            ).hexdigest()[:12])
            items.append({
                "task": "tf2_translation",
                "item_id": item_id,
                "fable": entry.get("fable", ""),
                "translated_fable": entry["translated_fable"],
                "translator_model": entry.get("translator", entry.get("model", "unknown")),
            })

    save_jsonl(items, os.path.join(output_dir, "samples.jsonl"))
    logger.info("TF2: exported %d translation pairs from %s", len(items), data_dir)


def export_tf3(data_dir: str, output_dir: str = "data/tf3"):
    """Export TF3 Romanian generation samples."""
    items = []
    for fpath in sorted(Path(data_dir).glob("**/*.txt")):
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        item_id = hashlib.md5(text.encode()).hexdigest()[:12]
        items.append({
            "task": "tf3_generation",
            "item_id": item_id,
            "fable": text,
            "prompt": "",
            "generator_model": fpath.parent.name,
        })

    if not items:
        for fpath in sorted(Path(data_dir).glob("**/*.jsonl")):
            for entry in load_jsonl(str(fpath)):
                fable = entry.get("fable", entry.get("text", ""))
                if not fable:
                    continue
                item_id = entry.get("hash", hashlib.md5(fable.encode()).hexdigest()[:12])
                items.append({
                    "task": "tf3_generation",
                    "item_id": item_id,
                    "fable": fable,
                    "prompt": entry.get("prompt", ""),
                    "generator_model": entry.get("model", "tf3"),
                })

    save_jsonl(items, os.path.join(output_dir, "samples.jsonl"))
    logger.info("TF3: exported %d items from %s", len(items), data_dir)


def main():
    parser = argparse.ArgumentParser(description="Export TF evaluation samples")
    parser.add_argument("--tf1-dir", help="Path to TF1 fables directory")
    parser.add_argument("--tf2-dir", help="Path to TF2 data directory")
    parser.add_argument("--tf3-dir", help="Path to TF3 evaluation artifacts")
    args = parser.parse_args()

    if args.tf1_dir:
        export_tf1(args.tf1_dir)
    if args.tf2_dir:
        export_tf2(args.tf2_dir)
    if args.tf3_dir:
        export_tf3(args.tf3_dir)

    if not any([args.tf1_dir, args.tf2_dir, args.tf3_dir]):
        parser.print_help()


if __name__ == "__main__":
    main()
