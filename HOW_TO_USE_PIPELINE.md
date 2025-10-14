# How to Use the SWE-bench Pipeline

This guide explains how to use `run_pipeline.py` to run the complete SWE-bench evaluation pipeline.

## Overview

The pipeline has 6 stages:

1. **Stage 1**: Generate patches using mini-swe-agent
2. **Stage 2**: Create SWE-bench prediction files from patches
3. **Stage 3**: Run evaluation using the SWE-bench harness
4. **Stage 4**: Run security scans (Bandit, Semgrep, CodeQL) before/after applying patches
5. **Stage 5**: Run consistency checks across multiple runs
6. **Stage 6**: Aggregate run-level results (JSON + CSV)

> **Windows users:** Run the pipeline from Ubuntu WSL so the CodeQL CLI and Docker (if needed) work as expected. Follow the Linux setup instructions inside WSL, including `codeql pack download codeql/python-queries`.

## Important: How the Pipeline Works

### Directory Structure and Timestamps

Each time you run the pipeline, it creates a **NEW timestamped directory**:

```
output/
└── <model-name>/
    └── <timestamp>/             # NEW directory created each run
        ├── config.json
        ├── stage1_summary.json
        ├── stage2_summary.json
        ├── stage3_summary.json
        ├── stage4_summary.json
        ├── stage5_summary.json
        ├── stage6_summary.json
        ├── predictions_all_run1.json
        ├── evaluation_results_run1.json
        ├── ...
        ├── django__django-10914/
        │   ├── run_1/
        │   │   ├── patch.diff
        │   │   ├── prediction.json
        │   │   ├── evaluation.json
        │   │   ├── trajectory.json
        │   │   └── metadata.json
        │   ├── run_2/
        │   │   └── ...
        │   └── consistency.json
        └── results.csv
```

**Key Points:**
- Each run gets a **unique timestamp** (e.g., `20251009_1830`)
- Multiple runs (`--num-runs`) create `run_<N>` subdirectories per instance
- Stage 1 **creates** the directory and runs patch generation
- Later stages reuse the same directory; keep the path handy
- `run_pipeline.py --stages ...` always creates a new timestamped folder—use individual scripts to continue an existing run
- Stage 4 clones repositories from GitHub and requires the CodeQL CLI plus the `codeql/python-queries` pack

## Usage Patterns

### Pattern 1: Run All Stages at Once (Recommended)

```bash
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances django__django-10914 \
    --num-runs 3 \
    --stages all
```

**What happens:**
1. Creates directory: `output/gemini/gemini-2.5-pro/20251009_1830/`
2. Runs all 6 stages sequentially (including security + consistency)
3. Produces per-run outputs in `run_1/`, `run_2/`, `run_3/`

**When to use:** When you want a complete run from start to finish.

---

### Pattern 2: Run Only Stage 1 (Patch Generation)

```bash
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances django__django-10914 \
    --stages 1
```

**What happens:**
1. Creates directory: `output/gemini/gemini-2.5-pro/20251009_1830/`
2. Generates patches
3. Stops after Stage 1

**When to use:** When you want to inspect patches before running evaluation.

---

### Pattern 3: Continue from Stage 1 (Manual Approach)

If you ran only Stage 1 and want to continue, you **must** use the individual stage scripts with the exact directory path:

#### Step 1: Find your run directory

```bash
ls -lt output/gemini/gemini-2.5-pro/
```

Output:
```
drwxr-xr-x  5 user  staff  160 Oct  9 18:30 20251009_1830
drwxr-xr-x  5 user  staff  160 Oct  9 17:45 20251009_1745
```

#### Step 2: Run remaining stages with that directory

```bash
# Replace 20251009_1830 with YOUR actual timestamp
RUN_DIR="output/gemini/gemini-2.5-pro/20251009_1830"

# Stage 2: Create predictions
python pipeline_2_create_predictions.py $RUN_DIR

# Stage 3: Run evaluation
python pipeline_3_run_evaluation.py $RUN_DIR

# Stage 4: Security scan (requires CodeQL CLI + codeql/python-queries pack)
python pipeline_4_security_scan.py $RUN_DIR

# Stage 5: Consistency check
python pipeline_5_consistency_check.py $RUN_DIR

# Stage 6: Aggregate results
python pipeline_6_aggregate_results.py $RUN_DIR
```

**Why this way:**
- `run_pipeline.py` always creates a **new timestamp**
- Stages 2-6 need the **exact directory** from Stage 1
- You must use the individual stage scripts directly

---

## Command-Line Options

### Required Arguments

```bash
--instances INSTANCE_IDS [INSTANCE_IDS ...]
```
One or more SWE-bench instance IDs to process.

**Example:**
```bash
--instances django__django-10914
--instances django__django-10914 django__django-10097
```

---

### Model Configuration

```bash
--model MODEL_NAME
```
The LLM model to use (default: `claude-sonnet-4-20250514`).

**Examples:**
```bash
--model claude-sonnet-4-20250514        # Anthropic Claude
--model gpt-4-turbo                     # OpenAI GPT
--model gemini/gemini-2.5-pro           # Google Gemini
--model openrouter/anthropic/claude-sonnet-4  # Via OpenRouter
```

---

### Optional Model Parameters

```bash
--temperature FLOAT
```
Model temperature (0.0-1.0). Lower = more deterministic.

**Examples:**
```bash
--temperature 0.0    # Fully deterministic
--temperature 0.3    # Slightly creative (good default)
--temperature 0.7    # More creative
```

```bash
--top-p FLOAT
```
Top-p nucleus sampling (0.0-1.0).

```bash
--custom-config PATH
```
Path to custom mini-swe-agent YAML config file.

---

### Pipeline Control

```bash
--stages {1,2,3,4,5,6,all} [{1,2,3,4,5,6,all} ...]
```
Which stages to run (default: `all`).

**Examples:**
```bash
--stages all               # Run all 6 stages
--stages 1                 # Run only Stage 1
--stages 1 2               # Run Stages 1 and 2
--stages 2 3 4 5 6         # ⚠️ Will fail - creates new directory!
```

**⚠️ Warning:** You cannot use `--stages 2 3 4 5 6` on an existing run because it creates a new timestamped directory.

```bash
--num-runs INT
```
How many independent runs to generate per instance (default: `1`). Each run gets its own `run_<N>` subdirectory and contributes to consistency checks.

---

### Dataset Configuration

```bash
--dataset DATASET_NAME
```
SWE-bench dataset to use (default: `princeton-nlp/SWE-bench_Lite`).

```bash
--split SPLIT_NAME
```
Dataset split to use (default: `test`).

```bash
--output-dir DIR
```
Base output directory (default: `output`).

```bash
--max-tokens INT
```
Maximum tokens to generate (default: `64000`).

---

## Complete Examples

### Example 1: Single Instance, All Stages

```bash
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances django__django-10914 \
    --num-runs 3 \
    --stages all
```

**Result:** Complete evaluation in `output/gemini/gemini-2.5-pro/<timestamp>/`

---

### Example 2: Multiple Instances with Single Run

```bash
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances django__django-10914 django__django-10097 django__django-11099 \
    --num-runs 1
```

**Result:** Evaluates 3 instances with 1 run each (default `--num-runs 1`) in `output/gemini/gemini-2.5-pro/<timestamp>/`

---

### Example 3: Multiple Instances with Multiple Runs and Temperature

```bash
python run_pipeline.py \
    --model claude-sonnet-4-20250514 \
    --instances django__django-10914 django__django-10097 django__django-11099 \
    --num-runs 2 \
    --temperature 0.3 \
    --stages all
```

**Result:** Evaluates 3 instances with 2 runs each for consistency analysis.

---

### Example 5: Stage 1 Only for Quick Testing

```bash
python run_pipeline.py \
    --model gpt-4-turbo \
    --instances django__django-10914 \
    --temperature 0.2 \
    --stages 1
```

**Then continue manually:**
```bash
# Find the directory
ls -lt output/gpt-4-turbo/

# Use the latest timestamp
python pipeline_2_create_predictions.py output/gpt-4-turbo/20251009_1830
python pipeline_3_run_evaluation.py output/gpt-4-turbo/20251009_1830
python pipeline_4_security_scan.py output/gpt-4-turbo/20251009_1830
python pipeline_5_consistency_check.py output/gpt-4-turbo/20251009_1830
python pipeline_6_aggregate_results.py output/gpt-4-turbo/20251009_1830
```

---

### Example 6: Custom Config

```bash
python run_pipeline.py \
    --model some-custom-model \
    --instances django__django-10914 \
    --num-runs 4 \
    --custom-config configs/my_model.yaml \
    --stages all
```

---

## Understanding the Output

### Output Directory Structure

```
output/
└── gemini/gemini-2.5-pro/
    └── 20251009_1830/                      # Timestamped run directory
        ├── config.json                     # Pipeline configuration
        ├── stage1_summary.json             # Stage 1 results
        ├── stage2_summary.json             # Stage 2 results
        ├── stage3_summary.json             # Stage 3 results
        ├── stage4_summary.json             # Stage 4 results
        ├── stage5_summary.json             # Stage 5 results
        ├── stage6_summary.json             # Stage 6 results
        ├── predictions_all_run1.json       # Predictions for run 1
        ├── evaluation_results_run1.json    # Evaluation for run 1
        ├── run_summary.json                # Aggregated summary
        ├── results.csv                     # CSV export
        ├── django__django-10914/           # Instance directory
        │   ├── run_1/
        │   │   ├── patch.diff
        │   │   ├── prediction.json
        │   │   ├── evaluation.json
        │   │   ├── trajectory.json
        │   │   ├── metadata.json
        │   │   └── security_risk_score.json
        │   ├── run_2/
        │   │   └── ...
        │   └── consistency.json
        └── ...
```

### Key Files

| File | Created By | Contains |
|------|------------|----------|
| `config.json` | Stage 1 | Pipeline configuration |
| `patch.diff` (per run) | Stage 1 | The actual code fix |
| `trajectory.json` (per run) | Stage 1 | Full agent execution trace |
| `metadata.json` (per run) | Stage 1 | Cost, API calls, timing |
| `prediction.json` (per run) | Stage 2 | SWE-bench prediction format |
| `predictions_all_runN.json` | Stage 2 | All predictions for run `N` |
| `evaluation.json` (per run) | Stage 3 | Test pass/fail results |
| `evaluation_results_runN.json` | Stage 3 | Official SWE-bench evaluation output |
| `security_risk_score.json` (per run) | Stage 4 | Bandit/Semgrep/CodeQL comparison |
| `consistency.json` | Stage 5 | Cross-run consistency scores |
| `run_summary.json` | Stage 6 | Aggregated metrics across runs |
| `results.csv` | Stage 6 | CSV export for analysis |

---

## Common Workflows

### Workflow 1: Quick Test, Then Full Run

```bash
# Step 1: Test with one instance
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances django__django-10914 \
    --stages 1

# Step 2: Check the patch
cat output/gemini/gemini-2.5-pro/*/django__django-10914/run_1/patch.diff

# Step 3: If good, run full evaluation on that directory
RUN_DIR=$(ls -t output/gemini/gemini-2.5-pro/ | head -1)
python pipeline_2_create_predictions.py output/gemini/gemini-2.5-pro/$RUN_DIR
python pipeline_3_run_evaluation.py output/gemini/gemini-2.5-pro/$RUN_DIR
python pipeline_4_security_scan.py output/gemini/gemini-2.5-pro/$RUN_DIR
python pipeline_5_consistency_check.py output/gemini/gemini-2.5-pro/$RUN_DIR
python pipeline_6_aggregate_results.py output/gemini/gemini-2.5-pro/$RUN_DIR
```

---

### Workflow 2: Batch Processing

```bash
# Create list of instances
cat > instances.txt << EOF
django__django-10914
django__django-10097
django__django-11099
EOF

# Run all at once
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances $(cat instances.txt) \
    --num-runs 2 \
    --stages all
```

---

### Workflow 3: Model Comparison

```bash
# Run same instances with different models
INSTANCES="django__django-10914 django__django-10097"

python run_pipeline.py --model claude-sonnet-4-20250514 --instances $INSTANCES --num-runs 2 --stages all
python run_pipeline.py --model gpt-4-turbo --instances $INSTANCES --num-runs 2 --stages all
python run_pipeline.py --model gemini/gemini-2.5-pro --instances $INSTANCES --num-runs 2 --stages all

# Compare results
ls -la output/*/*/results.csv
```

---

## Troubleshooting

### Problem: "Config not found" in Stage 2/3/4/5/6

**Cause:** You're trying to run `run_pipeline.py --stages ...` on an existing run, which creates a new timestamped directory without the original `config.json`.

**Solution:** Use the individual stage scripts with the existing directory:
```bash
RUN_DIR="output/gemini/gemini-2.5-pro/20251009_1830"
python pipeline_2_create_predictions.py $RUN_DIR
python pipeline_3_run_evaluation.py $RUN_DIR
python pipeline_4_security_scan.py $RUN_DIR
python pipeline_5_consistency_check.py $RUN_DIR
python pipeline_6_aggregate_results.py $RUN_DIR
```

---

### Problem: CodeQL scan fails (`codeql` not found or query pack missing)

**Cause:** The CodeQL CLI is not installed or the standard Python query pack (`codeql/python-queries`) has not been downloaded.

**Solution:**
```bash
# macOS
brew install --cask codeql
codeql pack download codeql/python-queries

# Linux / Windows via Ubuntu WSL
wget https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.23.2/codeql-bundle-linux64.tar.gz
sudo mkdir -p /usr/local/share/codeql
sudo chown $USER /usr/local/share/codeql
tar xf codeql-bundle-linux64.tar.gz -C /usr/local/share
sudo ln -s /usr/local/share/codeql/codeql /usr/local/bin/codeql
codeql pack download codeql/python-queries
```

---

### Problem: "API key not set"

**Cause:** Missing API key for the model provider.

**Solution:** Set the required API key:
```bash
# For Gemini
export GEMINI_API_KEY=your-key-here

# For Claude
export ANTHROPIC_API_KEY=your-key-here

# For GPT
export OPENAI_API_KEY=your-key-here

# Or add to .env file
echo "GEMINI_API_KEY=your-key-here" >> .env
```

---

### Problem: Can't find latest run directory

**Solution:** Use this command to get the latest run:
```bash
# Get latest run for a specific model
ls -t output/gemini/gemini-2.5-pro/ | head -1

# Or set it as a variable
RUN_DIR="output/gemini/gemini-2.5-pro/$(ls -t output/gemini/gemini-2.5-pro/ | head -1)"
echo $RUN_DIR
```

---

## Quick Reference

### Run everything at once:
```bash
python run_pipeline.py --model <MODEL> --instances <ID> --num-runs 2 --stages all
```

### Run Stage 1 only:
```bash
python run_pipeline.py --model <MODEL> --instances <ID> --stages 1
```

### Continue from Stage 1:
```bash
# Find directory
ls -lt output/<model-name>/

# Run remaining stages
python pipeline_2_create_predictions.py output/<model>/<timestamp>
python pipeline_3_run_evaluation.py output/<model>/<timestamp>
python pipeline_4_security_scan.py output/<model>/<timestamp>
python pipeline_5_consistency_check.py output/<model>/<timestamp>
python pipeline_6_aggregate_results.py output/<model>/<timestamp>
```

### Check results:
```bash
# View summary
cat output/<model>/<timestamp>/run_summary.json

# View CSV
cat output/<model>/<timestamp>/results.csv

# View patch
cat output/<model>/<timestamp>/<instance>/run_1/patch.diff
```

---

## See Also

- [Multi-Model Guide](features/multi-model-support/MULTI_MODEL_GUIDE.md) - How to use different AI providers
- [OpenRouter Setup](features/multi-model-support/OPENROUTER_SETUP.md) - Using OpenRouter for unified API access
- [Example Scripts](examples/) - Ready-to-use example scripts for each provider
