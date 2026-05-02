# Open-Weight Judges

Evaluation framework for studying **open-weight, multi-family LLM judge panels** across synthetic text evaluation tasks.

This project accompanies a research paper on **cross-task transferability of family-diverse open-weight judge panels for synthetic text evaluation** (in preparation). It provides tools for running multi-judge panels, computing inter-judge agreement, conducting bias audits, and comparing open-weight panels against proprietary baselines.

## Features

- **Multi-judge panels** -- run N judges from different model families on evaluation items, with resume support and sequential model loading for memory-constrained hardware
- **Three evaluation tasks** -- English fable generation (TF1), English-Romanian literary translation (TF2), Romanian native generation (TF3), each with task-specific rubrics and JSON schemas
- **Agreement metrics** -- Krippendorff's alpha, weighted Cohen's kappa, Gwet's AC2, Spearman/Kendall rank correlation
- **Aggregation methods** -- median/mean across judges, win-rate for system-level ranking
- **Bias audit suite** -- position sensitivity, length bias, family preference, repeated-run stability
- **Ollama + OpenRouter support** -- local open-weight models via Ollama, proprietary baselines via OpenRouter API

## Repository Structure

```
judges/                     # Core framework
├── judge.py                # Single-judge abstraction (retry, validation, token tracking)
├── panel.py                # Multi-judge orchestration with resume
├── config.py               # YAML configuration loader
├── aggregation.py          # Score aggregation (median, mean, win-rate)
├── agreement.py            # Inter-judge agreement metrics
├── rubrics/                # Task-specific evaluation rubrics
│   ├── tf1_generation.py   # English fable generation (4-dim, 1-10 scale)
│   ├── tf2_translation.py  # EN-RO translation quality (5-dim, 1-5 scale)
│   └── tf3_generation.py   # Romanian generation quality (5-dim, 1-5 scale)
└── bias/                   # Bias audit modules
    ├── position.py         # Order-swap sensitivity
    ├── length.py           # Length-controlled scoring
    ├── family.py           # Same-family preference detection
    └── stability.py        # Repeated-run reliability

scripts/                    # CLI entry points
├── run_panel.py            # Run judge panel on a task
├── run_bias_audit.py       # Run bias audit suite
├── analyze_agreement.py    # Compute agreement statistics
└── export_samples.py       # Export TF1/TF2/TF3 evaluation samples

conf/                       # Configuration
├── judges.yaml             # Judge model definitions (panel, baselines)
└── tasks.yaml              # Task and rubric mappings
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) for local open-weight model inference
- An [OpenRouter](https://openrouter.ai/) API key for proprietary baseline comparisons (optional)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and set your endpoints:

```bash
cp .env.example .env
```

Edit `.env` to point to your Ollama instance and (optionally) add your OpenRouter API key:

```
OLLAMA_BASE_URL=http://localhost:11434/v1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your-key-here
```

## Usage

```bash
# Export evaluation samples from TF1/TF2/TF3 repos
python scripts/export_samples.py --tf1-dir /path/to/tinyfabulist-tf1/data/fables

# Run the judge panel on a task
python scripts/run_panel.py --task tf1_generation

# Run proprietary baselines for comparison
python scripts/run_panel.py --task tf1_generation --group proprietary_baselines

# Analyze inter-judge agreement
python scripts/analyze_agreement.py --task tf1_generation --results-dir artifacts/<run>

# Run bias audits
python scripts/run_bias_audit.py --task tf1_generation --results-dir artifacts/<run> --audit all
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

## Related

- **Paper:** In preparation
- **TF1 dataset:** [klusai/ds-tf1-en-3m](https://huggingface.co/datasets/klusai/ds-tf1-en-3m) -- 3M English synthetic fables
- **TF2 dataset:** [klusai/ds-tf2-en-ro-3m](https://huggingface.co/datasets/klusai/ds-tf2-en-ro-3m) -- 3M English-Romanian parallel fables
- **TF3 dataset:** [klusai/ds-tf3-ro-3m](https://huggingface.co/datasets/klusai/ds-tf3-ro-3m) -- 3M Romanian-native fables

## License

[MIT](LICENSE)
