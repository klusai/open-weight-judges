#!/usr/bin/env python3
"""Generate all results tables and figures for the paper.

Reads all artifacts and produces CSV tables + LaTeX fragments.

Usage:
    python scripts/generate_paper_tables.py --output-dir artifacts/paper_tables
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges.panel import load_items
from judges.aggregation import median_aggregate, system_level_means
from judges.agreement import (
    krippendorff_alpha,
    pairwise_weighted_kappa,
    agreement_summary,
    spearman_correlation,
    kendall_tau,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_judge_results(results_dir, task_name):
    scores_by_judge = {}
    for f in sorted(Path(results_dir).glob(f"{task_name}_*.jsonl")):
        items = load_items(str(f))
        if items:
            judge = items[0].get("judge_model", f.stem)
            scores_by_judge[judge] = items
    return scores_by_judge


def extract_score(scores, dim):
    val = scores.get(dim)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def table_panel_agreement(output_dir):
    """Table 1: Panel internal agreement per task."""
    tasks = {
        "tf1_generation": {
            "dir": "artifacts/tf1_panel",
            "dims": ["grammar", "creativity", "moral_clarity", "adherence_to_prompt"],
        },
        "tf2_translation": {
            "dir": "artifacts/tf2_panel",
            "dims": ["accuracy", "fluency", "coherence", "style", "cultural_pragmatic"],
        },
        "tf3_generation": {
            "dir": "artifacts/tf3_panel",
            "dims": ["grammar", "creativity", "coherence", "moral_clarity", "adherence"],
        },
    }

    rows = []
    for task, cfg in tasks.items():
        sbj = load_judge_results(cfg["dir"], task)
        if len(sbj) < 2:
            continue
        summary = agreement_summary(sbj, cfg["dims"])
        for _, row in summary.iterrows():
            rows.append({
                "task": task,
                "dimension": row["dimension"],
                "krippendorff_alpha": row["krippendorff_alpha"],
                "mean_weighted_kappa": row["mean_weighted_kappa"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "table1_panel_agreement.csv"), index=False)
    logger.info("Table 1: Panel agreement saved (%d rows)", len(df))
    return df


def table_panel_vs_proprietary(output_dir):
    """Table 2: Panel median vs proprietary baseline correlation."""
    configs = [
        ("tf1_generation", "artifacts/tf1_panel", "artifacts/tf1_proprietary",
         ["grammar", "creativity", "moral_clarity", "adherence_to_prompt"]),
        ("tf2_translation", "artifacts/tf2_panel", "artifacts/tf2_proprietary",
         ["accuracy", "fluency", "coherence", "style", "cultural_pragmatic"]),
        ("tf3_generation", "artifacts/tf3_panel", "artifacts/tf3_proprietary",
         ["grammar", "creativity", "coherence", "moral_clarity", "adherence"]),
    ]

    rows = []
    for task, panel_dir, prop_dir, dims in configs:
        panel_sbj = load_judge_results(panel_dir, task)
        prop_sbj = load_judge_results(prop_dir, task)

        if not panel_sbj or not prop_sbj:
            continue

        panel_agg = median_aggregate(panel_sbj, dims)

        for prop_name, prop_items in prop_sbj.items():
            prop_map = {}
            for item in prop_items:
                for dim in dims:
                    val = extract_score(item.get("scores", {}), dim)
                    if val is not None:
                        prop_map.setdefault(dim, {})[item["item_id"]] = val

            for dim in dims:
                col = f"{dim}_median"
                if col not in panel_agg.columns:
                    continue
                panel_scores = {}
                for _, r in panel_agg.iterrows():
                    if r[col] is not None and not np.isnan(r[col]):
                        panel_scores[r["item_id"]] = r[col]

                dim_prop = prop_map.get(dim, {})
                common = sorted(set(panel_scores.keys()) & set(dim_prop.keys()))
                if len(common) < 5:
                    continue

                p_vals = [panel_scores[k] for k in common]
                b_vals = [dim_prop[k] for k in common]
                rho, p = stats.spearmanr(p_vals, b_vals)

                rows.append({
                    "task": task,
                    "proprietary": prop_name,
                    "dimension": dim,
                    "spearman_rho": round(rho, 4),
                    "p_value": round(p, 6),
                    "n": len(common),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "table2_panel_vs_proprietary.csv"), index=False)
    logger.info("Table 2: Panel vs proprietary saved (%d rows)", len(df))
    return df


def table_system_rankings(output_dir):
    """Table 3: TF1 system-level rankings across judges."""
    dims = ["grammar", "creativity", "moral_clarity", "adherence_to_prompt"]

    sources = {
        "panel_median": ("artifacts/tf1_panel", True),
        "o4-mini": ("artifacts/tf1_proprietary", False),
        "selene-mini": ("artifacts/tf1_selene", False),
    }

    rankings = {}
    for source_name, (source_dir, is_panel) in sources.items():
        sbj = load_judge_results(source_dir, "tf1_generation")
        if not sbj:
            continue

        if is_panel:
            sys_df = system_level_means(sbj, dims, "generator_model")
        else:
            judge_name = list(sbj.keys())[0]
            items = sbj[judge_name]
            by_model = {}
            for item in items:
                m = item.get("generator_model", "unknown")
                if m not in by_model:
                    by_model[m] = {d: [] for d in dims}
                for d in dims:
                    val = extract_score(item.get("scores", {}), d)
                    if val is not None:
                        by_model[m][d].append(val)

            sys_rows = []
            for m, scores in by_model.items():
                row = {"system": m}
                for d in dims:
                    row[d] = np.mean(scores[d]) if scores[d] else None
                sys_rows.append(row)
            sys_df = pd.DataFrame(sys_rows)

        if sys_df.empty:
            continue

        sys_df["composite"] = sys_df[dims].mean(axis=1)
        sys_df = sys_df.sort_values("composite", ascending=False).reset_index(drop=True)
        sys_df["rank"] = range(1, len(sys_df) + 1)
        rankings[source_name] = sys_df[["system", "composite", "rank"]].rename(
            columns={"composite": f"composite_{source_name}", "rank": f"rank_{source_name}"}
        )

    if len(rankings) >= 2:
        merged = None
        for name, df in rankings.items():
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="system", how="outer")

        merged.to_csv(os.path.join(output_dir, "table3_system_rankings.csv"), index=False)
        logger.info("Table 3: System rankings saved")

        rank_cols = [c for c in merged.columns if c.startswith("rank_")]
        if len(rank_cols) >= 2:
            print("\n=== System-Level Rank Correlations ===")
            for i, c1 in enumerate(rank_cols):
                for c2 in rank_cols[i + 1:]:
                    valid = merged.dropna(subset=[c1, c2])
                    if len(valid) >= 3:
                        tau, p_tau = stats.kendalltau(valid[c1], valid[c2])
                        rho, p_rho = stats.spearmanr(valid[c1], valid[c2])
                        print(f"  {c1} vs {c2}: tau={tau:.3f} (p={p_tau:.3f}), rho={rho:.3f} (p={p_rho:.3f})")

        return merged
    return None


def table_bias_audit(output_dir):
    """Table 4: Bias audit summary."""
    parts = []

    pos_path = "artifacts/bias_audit/tf1_generation_position/tf1_generation_position_summary.csv"
    if os.path.exists(pos_path):
        pos = pd.read_csv(pos_path)
        pos["audit"] = "position"
        parts.append(pos[["audit", "judge", "dimension", "exact_match_rate", "mean_abs_delta"]])

    stab_path = "artifacts/bias_audit/tf1_generation_stability/tf1_generation_stability_summary.csv"
    if os.path.exists(stab_path):
        stab = pd.read_csv(stab_path)
        stab["audit"] = "stability"
        stab["mean_abs_delta"] = stab["mean_abs_delta"]
        parts.append(stab[["audit", "judge", "dimension", "exact_match_rate", "mean_abs_delta"]])

    len_path = "artifacts/bias_audit/tf1_generation_length/tf1_generation_length_summary.csv"
    if os.path.exists(len_path):
        length = pd.read_csv(len_path)
        corr_rows = length[length["length_bin"] == "correlation"]
        if not corr_rows.empty:
            corr_rows = corr_rows.rename(columns={"mean_score": "spearman_rho", "std_score": "p_value"})
            corr_rows["audit"] = "length"
            corr_rows["exact_match_rate"] = None
            corr_rows["mean_abs_delta"] = corr_rows["spearman_rho"]
            parts.append(corr_rows[["audit", "judge", "dimension", "exact_match_rate", "mean_abs_delta"]])

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "table4_bias_audit.csv"), index=False)
        logger.info("Table 4: Bias audit saved (%d rows)", len(combined))
        return combined
    return None


def table_cost(output_dir):
    """Table 5: Cost comparison."""
    rows = []
    for task_dir, task_name in [
        ("artifacts/tf2_panel", "tf2_translation"),
        ("artifacts/tf3_panel", "tf3_generation"),
    ]:
        summary_path = os.path.join(task_dir, f"{task_name}_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summaries = json.load(f)
            for s in summaries:
                rows.append({
                    "task": task_name,
                    "judge": s["judge"],
                    "role": "panel",
                    "items": s["evaluated"],
                    "input_tokens": s.get("total_input_tokens", 0),
                    "output_tokens": s.get("total_output_tokens", 0),
                    "elapsed_seconds": s.get("elapsed_seconds", 0),
                    "endpoint": "ollama (local)",
                    "estimated_cost_usd": 0.0,
                })

    for prop_dir in ["artifacts/tf2_proprietary", "artifacts/tf3_proprietary"]:
        for f in Path(prop_dir).glob("*.jsonl"):
            items = load_items(str(f))
            if not items:
                continue
            judge = items[0].get("judge_model", f.stem)
            task = items[0].get("task", "unknown")
            total_in = sum(i.get("input_tokens", 0) for i in items)
            total_out = sum(i.get("output_tokens", 0) for i in items)
            total_latency = sum(i.get("latency_ms", 0) for i in items) / 1000
            rows.append({
                "task": task,
                "judge": judge,
                "role": "proprietary",
                "items": len(items),
                "input_tokens": total_in,
                "output_tokens": total_out,
                "elapsed_seconds": round(total_latency, 1),
                "endpoint": "openrouter",
                "estimated_cost_usd": round((total_in * 1.1 + total_out * 4.4) / 1_000_000, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "table5_cost.csv"), index=False)
    logger.info("Table 5: Cost saved (%d rows)", len(df))
    return df


def table_selene_comparison(output_dir):
    """Table 6: Selene Mini vs panel vs proprietary."""
    configs = [
        ("tf1_generation", "artifacts/tf1_selene", "artifacts/tf1_panel", "artifacts/tf1_proprietary",
         ["grammar", "creativity", "moral_clarity", "adherence_to_prompt"]),
        ("tf2_translation", "artifacts/tf2_selene", "artifacts/tf2_panel", "artifacts/tf2_proprietary",
         ["accuracy", "fluency", "coherence", "style", "cultural_pragmatic"]),
        ("tf3_generation", "artifacts/tf3_selene", "artifacts/tf3_panel", "artifacts/tf3_proprietary",
         ["grammar", "creativity", "coherence", "moral_clarity", "adherence"]),
    ]

    rows = []
    for task, selene_dir, panel_dir, prop_dir, dims in configs:
        selene_sbj = load_judge_results(selene_dir, task)
        panel_sbj = load_judge_results(panel_dir, task)
        prop_sbj = load_judge_results(prop_dir, task)

        if not selene_sbj:
            continue

        selene_name = list(selene_sbj.keys())[0]
        selene_items = selene_sbj[selene_name]
        selene_map = {item["item_id"]: item for item in selene_items}

        panel_agg = median_aggregate(panel_sbj, dims)
        panel_map = {r["item_id"]: r for _, r in panel_agg.iterrows()}

        for dim in dims:
            selene_scores = {}
            for iid, item in selene_map.items():
                val = extract_score(item.get("scores", {}), dim)
                if val is not None:
                    selene_scores[iid] = val

            panel_scores = {}
            for iid, r in panel_map.items():
                col = f"{dim}_median"
                if col in r and r[col] is not None and not np.isnan(r[col]):
                    panel_scores[iid] = r[col]

            common = sorted(set(selene_scores.keys()) & set(panel_scores.keys()))
            if len(common) >= 5:
                s_vals = [selene_scores[k] for k in common]
                p_vals = [panel_scores[k] for k in common]
                rho, p = stats.spearmanr(s_vals, p_vals)
                rows.append({
                    "task": task,
                    "comparison": "selene_vs_panel",
                    "dimension": dim,
                    "spearman_rho": round(rho, 4),
                    "p_value": round(p, 6),
                    "n": len(common),
                })

            for prop_name, prop_items in prop_sbj.items():
                prop_scores = {}
                for item in prop_items:
                    val = extract_score(item.get("scores", {}), dim)
                    if val is not None:
                        prop_scores[item["item_id"]] = val

                common2 = sorted(set(selene_scores.keys()) & set(prop_scores.keys()))
                if len(common2) >= 5:
                    s2 = [selene_scores[k] for k in common2]
                    p2 = [prop_scores[k] for k in common2]
                    rho2, pv2 = stats.spearmanr(s2, p2)
                    rows.append({
                        "task": task,
                        "comparison": f"selene_vs_{prop_name}",
                        "dimension": dim,
                        "spearman_rho": round(rho2, 4),
                        "p_value": round(pv2, 6),
                        "n": len(common2),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "table6_selene_comparison.csv"), index=False)
    logger.info("Table 6: Selene comparison saved (%d rows)", len(df))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/paper_tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING ALL PAPER TABLES")
    print("=" * 60)

    t1 = table_panel_agreement(args.output_dir)
    print("\n--- Table 1: Panel Agreement ---")
    print(t1.to_string(index=False))

    t2 = table_panel_vs_proprietary(args.output_dir)
    print("\n--- Table 2: Panel vs Proprietary ---")
    print(t2.to_string(index=False))

    t3 = table_system_rankings(args.output_dir)
    if t3 is not None:
        print("\n--- Table 3: System Rankings ---")
        print(t3.to_string(index=False))

    t4 = table_bias_audit(args.output_dir)
    if t4 is not None:
        print("\n--- Table 4: Bias Audit ---")
        print(t4.to_string(index=False))

    t5 = table_cost(args.output_dir)
    print("\n--- Table 5: Cost ---")
    print(t5.to_string(index=False))

    t6 = table_selene_comparison(args.output_dir)
    print("\n--- Table 6: Selene Comparison ---")
    print(t6.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"All tables saved to {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
