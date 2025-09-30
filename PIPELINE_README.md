# SWE-bench Evaluation Pipeline

Modular, extensible pipeline for evaluating AI models on SWE-bench tasks using mini-swe-agent.

## Overview

This pipeline automates the complete evaluation process:

1. **Generate Patches** - Use mini-swe-agent to solve SWE-bench instances
2. **Create Predictions** - Convert patches to SWE-bench prediction format
3. **Run Evaluation** - Execute official SWE-bench evaluation harness
4. **Aggregate Results** - Create comprehensive summaries and analysis

## Directory Structure

```
output/
├── claude-sonnet-4-20250514/
│   └── 20250930_0928/
│       ├── config.json                  # Run configuration
│       ├── stage1_summary.json          # Stage 1 summary
│       ├── stage2_summary.json          # Stage 2 summary
│       ├── stage3_summary.json          # Stage 3 summary
│       ├── run_summary.json             # Complete run summary
│       ├── results.csv                  # Results in CSV format
│       ├── predictions_all.json         # All predictions for SWE-bench
│       ├── evaluation_results.json      # Full evaluation results
│       ├── django__django-10914/
│       │   ├── patch.diff               # Generated code fix
│       │   ├── prediction.json          # SWE-bench prediction
│       │   ├── evaluation.json          # Evaluation result
│       │   ├── trajectory.json          # Agent execution trace
│       │   └── metadata.json            # Cost, time, API calls
│       └── django__django-11001/
│           └── ...
└── gpt-4o-2024-08-06/
    └── ...
```

## Quick Start

### Run Complete Pipeline

```bash
# Single instance
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914

# Multiple instances
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 django__django-11001 django__django-11019

# 10 instances from SWE-bench Lite
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 django__django-11001 django__django-11019 \
               django__django-11039 django__django-11049 django__django-11099 \
               django__django-11133 django__django-11179 django__django-11283 \
               django__django-11422
```

### Run Individual Stages

```bash
# Stage 1: Generate patches
python pipeline_1_generate_patches.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 django__django-11001

# Stage 2: Create predictions (requires Stage 1)
python pipeline_2_create_predictions.py \
  output/claude-sonnet-4-20250514/20250930_0928

# Stage 3: Run evaluation (requires Stage 2)
python pipeline_3_run_evaluation.py \
  output/claude-sonnet-4-20250514/20250930_0928

# Stage 4: Aggregate results (requires Stage 3)
python pipeline_4_aggregate_results.py \
  output/claude-sonnet-4-20250514/20250930_0928
```

### Run Specific Stages Only

```bash
# Re-run just evaluation and aggregation
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 \
  --stages 3 4
```

## Configuration

### Environment Setup

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=your_key_here
```

### Pipeline Configuration

Edit `pipeline_minisweagent_config.py`:

```python
@dataclass
class PipelineConfig:
    # Model configuration
    model_name: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_API_KEY"

    # Instance configuration
    instance_ids: List[str] = field(default_factory=lambda: ["django__django-10914"])
    swebench_dataset: str = "princeton-nlp/SWE-bench_Lite"
    swebench_split: str = "test"

    # Pipeline configuration
    max_tokens: int = 64000
    cost_limit: float = 3.0
    step_limit: int = 250
    max_workers: int = 1
    timeout: int = 600  # seconds per instance
```

## Output Files

### Per Instance

- **`patch.diff`** - The generated code fix in unified diff format
- **`prediction.json`** - SWE-bench prediction format for evaluation
- **`evaluation.json`** - Evaluation result (resolved/unresolved)
- **`trajectory.json`** - Complete mini-swe-agent execution trace
- **`metadata.json`** - Execution metrics (cost, time, API calls)

### Aggregated

- **`run_summary.json`** - Complete summary with all metrics
- **`results.csv`** - Results in CSV format for analysis
- **`predictions_all.json`** - All predictions in SWE-bench format
- **`evaluation_results.json`** - Official SWE-bench evaluation results

## Example Output

### run_summary.json

```json
{
  "run_info": {
    "model_name": "claude-sonnet-4-20250514",
    "timestamp": "20250930_0928",
    "swebench_dataset": "princeton-nlp/SWE-bench_Lite"
  },
  "overall_metrics": {
    "total_instances": 10,
    "resolved": 7,
    "unresolved": 3,
    "resolution_rate_percent": 70.0,
    "total_cost_usd": 2.45,
    "avg_cost_per_instance_usd": 0.245,
    "total_time_seconds": 450.3,
    "avg_time_per_instance_seconds": 45.0,
    "total_api_calls": 287
  },
  "resolved_instances": [
    "django__django-10914",
    "django__django-11001",
    ...
  ],
  "instance_results": [ ... ]
}
```

## Extending the Pipeline

### Adding a New Stage (e.g., Consistency Check)

1. **Create new stage script**: `pipeline_5_consistency_check.py`

```python
#!/usr/bin/env python3
"""
Stage 5: Consistency check

Runs consistency evaluation on generated patches.
"""

class ConsistencyChecker:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.config = self._load_config()

    def check_instance(self, instance_id: str) -> Dict:
        # Your consistency check logic here
        pass

    def check_all(self) -> Dict:
        # Iterate through instances
        pass
```

2. **Update master pipeline**: Add to `run_pipeline.py`

```python
# Stage 5: Consistency check
if "5" in stages_to_run:
    success = run_stage(
        "pipeline_5_consistency_check.py",
        [str(run_dir)]
    )
```

3. **Update output structure**: Add to instance directory

```
django__django-10914/
├── patch.diff
├── ...
└── consistency.json    # NEW: Consistency check results
```

### Supporting New Models

The pipeline is model-agnostic. To add a new model:

```bash
# OpenAI GPT-4
python run_pipeline.py \
  --model gpt-4-turbo-2024-04-09 \
  --instances django__django-10914

# Mistral
python run_pipeline.py \
  --model mistralai/Codestral-22B-v0.1 \
  --instances django__django-10914
```

Output will be organized under:
```
output/
├── gpt-4-turbo-2024-04-09/
│   └── 20250930_1045/
└── mistralai_Codestral-22B-v0.1/
    └── 20250930_1100/
```

## Advanced Usage

### Batch Processing

Create a script to run multiple models:

```python
#!/usr/bin/env python3
"""Run pipeline on multiple models."""

models = [
    "claude-sonnet-4-20250514",
    "gpt-4-turbo-2024-04-09",
    "claude-3-5-haiku-20241022",
]

instances = [
    "django__django-10914",
    "django__django-11001",
    # ... add more
]

for model in models:
    subprocess.run([
        "python", "run_pipeline.py",
        "--model", model,
        "--instances", *instances
    ])
```

### Comparing Results

```python
import json
from pathlib import Path

# Load results from multiple runs
runs = [
    "output/claude-sonnet-4-20250514/20250930_0928",
    "output/gpt-4-turbo-2024-04-09/20250930_1045",
]

for run_dir in runs:
    with open(Path(run_dir) / "run_summary.json") as f:
        summary = json.load(f)
        print(f"{summary['run_info']['model_name']}: "
              f"{summary['overall_metrics']['resolution_rate_percent']}% "
              f"(${summary['overall_metrics']['total_cost_usd']:.2f})")
```

## Troubleshooting

### Docker Not Running

```bash
# Start Docker
# On macOS with OrbStack:
orbstack start

# Check Docker is running
docker ps
```

### API Key Issues

```bash
# Check .env file exists
cat .env

# Verify key is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('ANTHROPIC_API_KEY') else 'NOT FOUND')"
```

### Pipeline Stage Failed

Each stage can be re-run independently:

```bash
# Re-run just the failed stage
python pipeline_3_run_evaluation.py output/claude-sonnet-4-20250514/20250930_0928
```

## Dependencies

```bash
pip install mini-swe-agent swebench python-dotenv
```

## License

MIT