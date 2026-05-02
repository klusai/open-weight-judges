---
license: mit
task_categories:
  - text-classification
language:
  - en
  - ro
tags:
  - llm-as-judge
  - evaluation
  - multi-judge-panel
  - bias-audit
  - synthetic-data
size_categories:
  - 1K<n<10K
---

# Open-Weight Judges Evaluation Dataset

Evaluation outputs from a study on **cross-task transferability of family-diverse open-weight judge panels for synthetic text evaluation**.

## Overview

This dataset contains ~6,180 judge evaluations across three synthetic-text tasks, produced by a three-member open-weight panel, proprietary baselines (GPT-o4-mini, Gemini 2.5 Flash), a specialized open evaluator (Atla Selene Mini), and four bias audits.

## Tasks

| Task | Items | Description | Scale |
|------|-------|-------------|-------|
| TF1 (English Generation) | 1,000 | English fable quality from 10 generator models | 1-10, 4 dimensions |
| TF2 (EN-RO Translation) | 96 | English-Romanian literary translation quality | 1-5, 5 dimensions |
| TF3 (Romanian Generation) | 100 | Romanian-native fable quality | 1-5, 5 dimensions |

## Panel Members

| Model | Params | Family | Role |
|-------|--------|--------|------|
| Granite 4.1 Dense | 30B | IBM | Primary |
| EXAONE 3.5 | 32B | LG AI Research | Primary |
| Granite 3.3 | 8B | IBM | Arbiter |

Panel members are deliberately from families disjoint from all generator models.

## Directory Structure

```
tf1_panel/          # Panel evaluations: 3 judges x 1000 items
tf2_panel/          # Panel evaluations: 3 judges x 96 items
tf3_panel/          # Panel evaluations: 3 judges x 100 items
tf1_proprietary/    # GPT-o4-mini on 1000 TF1 items
tf2_proprietary/    # GPT-o4-mini on 96 TF2 items
tf3_proprietary/    # GPT-o4-mini + Gemini Flash on 100 TF3 items each
tf1_selene/         # Selene Mini on 200 TF1 items
tf2_selene/         # Selene Mini on 96 TF2 items
tf3_selene/         # Selene Mini on 100 TF3 items
bias_audit/         # Position, stability, length, family audit results
paper_tables/       # Pre-computed analysis tables (CSV)
tf1_samples.jsonl   # Input fables (TF1)
tf2_samples.jsonl   # Input translation pairs (TF2)
tf3_samples.jsonl   # Input Romanian fables (TF3)
```

## Record Format

Panel and proprietary outputs use JSONL with one record per item:

```json
{
  "item_id": "...",
  "judge_model": "granite4.1:30b",
  "task": "tf1_generation",
  "scores": {"grammar": 8, "creativity": 6, ...},
  "input_tokens": 1714,
  "output_tokens": 164,
  "latency_ms": 27370.9,
  "timestamp": "2026-05-01T21:28:39"
}
```

TF2/TF3 scores use nested format: `{"accuracy": {"score": 5, "justification": "..."}}`.

## Key Findings

- **System-level rank correlation** between panel and GPT-o4-mini: rho=0.927 (p<0.001) on TF1's 10 generators
- **Item-level Krippendorff alpha** near zero across all tasks (score compression)
- **EXAONE 3.5** most position-stable (100% grammar match) and run-stable (91-100%)
- **Granite 4.1** shows significant length bias (rho=-0.27 for grammar)
- **Zero family-preference bias** by construction (disjoint panel-generator families)

## Citation

```bibtex
@article{nadas2026openjudges,
  title={Cross-Task Transferability of Family-Diverse Open-Weight Judge Panels for Synthetic Text Evaluation},
  author={Nada\c{s}, Mihai and Dio\c{s}an, Laura},
  year={2026},
  note={In preparation}
}
```

## Related

- **Code:** [klusai/open-weight-judges](https://github.com/klusai/open-weight-judges)
- **TF1 dataset:** [klusai/ds-tf1-en-3m](https://huggingface.co/datasets/klusai/ds-tf1-en-3m)
- **TF2 dataset:** [klusai/ds-tf2-en-ro-3m](https://huggingface.co/datasets/klusai/ds-tf2-en-ro-3m)
- **TF3 dataset:** [klusai/ds-tf3-ro-3m](https://huggingface.co/datasets/klusai/ds-tf3-ro-3m)
