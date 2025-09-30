# Pipeline Implementation Summary

## ✅ What Was Built

A complete, modular, and extensible SWE-bench evaluation pipeline with 4 stages:

### Pipeline Files Created

1. **`pipeline_minisweagent_config.py`** - Central configuration module
2. **`pipeline_1_generate_patches.py`** - Stage 1: Generate patches with mini-swe-agent
3. **`pipeline_2_create_predictions.py`** - Stage 2: Create SWE-bench predictions
4. **`pipeline_3_run_evaluation.py`** - Stage 3: Run official evaluation
5. **`pipeline_4_aggregate_results.py`** - Stage 4: Aggregate & analyze results
6. **`run_pipeline.py`** - Master pipeline runner
7. **`example_run.sh`** - Example usage script
8. **`PIPELINE_README.md`** - Complete documentation

## Directory Structure

```
output/
└── {model_name}/
    └── {timestamp}/
        ├── config.json
        ├── stage1_summary.json
        ├── stage2_summary.json
        ├── stage3_summary.json
        ├── run_summary.json
        ├── results.csv
        ├── predictions_all.json
        ├── evaluation_results.json
        └── {instance_id}/
            ├── patch.diff
            ├── prediction.json
            ├── evaluation.json
            ├── trajectory.json
            └── metadata.json
```

## Key Features

### ✅ Modular Design
- Each stage is independent
- Can run stages individually or together
- Easy to add new stages (e.g., consistency checks)

### ✅ Comprehensive Tracking
- **Per Instance**: Cost, time, API calls, patch size, status
- **Aggregated**: Overall metrics, success rates, summaries
- **Multiple Formats**: JSON for processing, CSV for analysis

### ✅ Model Agnostic
- Works with any model supported by mini-swe-agent
- Organizes output by model name
- Easy to compare multiple models

### ✅ Production Ready
- Error handling at each stage
- Detailed logging
- Configurable timeouts and limits
- Resume capability (re-run individual stages)

### ✅ Extensible
- Clean separation of concerns
- Well-documented code
- Easy to add new pipeline stages
- Configuration-driven

## Usage Examples

### Quick Test (1 instance)
```bash
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914
```

### Multiple Instances (10)
```bash
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 django__django-11001 django__django-11019 \
               django__django-11039 django__django-11049 django__django-11099 \
               django__django-11133 django__django-11179 django__django-11283 \
               django__django-11422
```

### Batch Run (100 instances)
```bash
# Create list of 100 SWE-bench Lite instances
python -c "
from datasets import load_dataset
ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
instances = [inst['instance_id'] for inst in ds][:100]
print(' '.join(instances))
" > instances_100.txt

# Run pipeline
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances $(cat instances_100.txt)
```

### Re-run Specific Stage
```bash
# If evaluation failed, re-run just that stage
python pipeline_3_run_evaluation.py \
  output/claude-sonnet-4-20250514/20250930_0928
```

## Output Files Explained

### instance_id/patch.diff
```diff
diff --git a/django/conf/global_settings.py b/django/conf/global_settings.py
--- a/django/conf/global_settings.py
+++ b/django/conf/global_settings.py
@@ -304,7 +304,7 @@ FILE_UPLOAD_TEMP_DIR = None
-FILE_UPLOAD_PERMISSIONS = None
+FILE_UPLOAD_PERMISSIONS = 0o644
```

### instance_id/metadata.json
```json
{
  "instance_id": "django__django-10914",
  "model_name": "claude-sonnet-4-20250514",
  "status": "completed",
  "cost": 0.2786,
  "api_calls": 39,
  "elapsed_time": 34.2,
  "exit_status": "Submitted"
}
```

### instance_id/evaluation.json
```json
{
  "instance_id": "django__django-10914",
  "status": "resolved",
  "resolved": true
}
```

### run_summary.json
```json
{
  "run_info": {
    "model_name": "claude-sonnet-4-20250514",
    "timestamp": "20250930_0928"
  },
  "overall_metrics": {
    "total_instances": 10,
    "resolved": 7,
    "resolution_rate_percent": 70.0,
    "total_cost_usd": 2.45,
    "avg_cost_per_instance_usd": 0.245
  },
  "resolved_instances": [...],
  "unresolved_instances": [...]
}
```

## Adding Consistency Checks (Future)

To add a consistency check stage:

1. **Create `pipeline_5_consistency_check.py`**:
```python
class ConsistencyChecker:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def check_instance(self, instance_id: str) -> Dict:
        # Load patch
        patch_path = self.run_dir / instance_id / "patch.diff"

        # Run consistency checks
        results = {
            "consistent": True,
            "issues": [],
            "score": 0.95
        }

        # Save results
        consistency_path = self.run_dir / instance_id / "consistency.json"
        with open(consistency_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
```

2. **Update `run_pipeline.py`**:
```python
# Stage 5: Consistency check
if "5" in stages_to_run:
    success = run_stage(
        "pipeline_5_consistency_check.py",
        [str(run_dir)]
    )
```

3. **Output structure**:
```
django__django-10914/
├── patch.diff
├── prediction.json
├── evaluation.json
├── trajectory.json
├── metadata.json
└── consistency.json    # NEW
```

## Comparing Multiple Models

```python
#!/usr/bin/env python3
"""Compare results across models."""

import json
from pathlib import Path
import pandas as pd

# Load results from different models
models = [
    "claude-sonnet-4-20250514",
    "gpt-4-turbo-2024-04-09",
    "claude-3-5-haiku-20241022",
]

results = []
for model in models:
    # Find latest run for each model
    model_dir = Path("output") / model
    latest_run = max(model_dir.iterdir(), key=lambda p: p.name)

    with open(latest_run / "run_summary.json") as f:
        summary = json.load(f)

    results.append({
        "model": model,
        "resolved": summary["overall_metrics"]["resolved"],
        "total": summary["overall_metrics"]["total_instances"],
        "rate": summary["overall_metrics"]["resolution_rate_percent"],
        "cost": summary["overall_metrics"]["total_cost_usd"],
        "time": summary["overall_metrics"]["total_time_seconds"],
    })

# Create comparison DataFrame
df = pd.DataFrame(results)
df = df.sort_values("rate", ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(df.to_string(index=False))
print("="*80)
```

## Performance Characteristics

Based on our test run (django__django-10914):

- **Time per instance**: ~35 seconds
- **Cost per instance**: ~$0.28 (Claude Sonnet 4)
- **API calls per instance**: ~40

**Estimated for 100 instances:**
- Time: ~58 minutes
- Cost: ~$28
- Total API calls: ~4,000

## Best Practices

1. **Start Small**: Test with 1-2 instances first
2. **Check Docker**: Ensure Docker is running before starting
3. **Monitor Costs**: Watch API usage for large runs
4. **Save Logs**: Pipeline logs everything for debugging
5. **Resume Failed Runs**: Re-run individual stages if needed

## Next Steps

1. ✅ Pipeline is ready to use
2. 🔄 Test with single instance
3. 📈 Scale to 10 instances
4. 🚀 Run full 100-instance evaluation
5. 🔍 Add consistency check stage (optional)

## Questions?

See `PIPELINE_README.md` for complete documentation.