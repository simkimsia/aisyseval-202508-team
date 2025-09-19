# Modal Setup for Large Model SWE-bench Evaluation

Complete guide for running your large models (gpt-oss-20b, qwen3-coder-30b, devstral-small-2507) on Modal with full Docker support.

## Why Modal for Your Use Case

✅ **A100 80GB GPUs** - Perfect for 20B-30B models
✅ **Full Docker support** - Real SWE-bench evaluation
✅ **Pay-per-second** - No idle costs
✅ **Auto-scaling** - Handle multiple instances
✅ **128GB RAM** - Load large models comfortably

## Prerequisites

- Modal account (free tier available)
- Your target models: `gpt-oss-20b`, `qwen3-coder-30b-a3b-instruct`, `devstral-small-2507`
- API keys if using Anthropic models

## Step 1: Install Modal

```bash
# Install Modal CLI
pip install modal

# Setup Modal account (follow prompts)
modal setup

# Verify installation
modal --help
```

## Step 2: Prepare Your Project

### 2.1 Ensure Files Are Ready
```bash
# Verify required files exist
ls -la flexible_pipeline.py  # Your main pipeline
ls -la modal_swebench.py     # Modal deployment script (created above)
```

### 2.2 Test Flexible Pipeline Locally (Optional)
```bash
# Quick local test with small model
python flexible_pipeline.py \
  --model "gpt2" \
  --instances "django__django-10097" \
  --max-tokens 30 \
  --device "cpu"
```

## Step 3: Deploy and Test on Modal

### 3.1 Test with Small Model First
```bash
# Test Modal setup with small model (faster/cheaper)
modal run modal_swebench.py \
  --model="microsoft/CodeGPT-small-py" \
  --instances="django__django-10097" \
  --max-tokens=50
```

**Expected output:**
```
🚀 Starting evaluation on Modal GPU
Model: microsoft/CodeGPT-small-py
✅ GPU available: NVIDIA A100-SXM4-80GB (80.0GB)
🐳 Docker daemon ready
📥 Loading model...
📚 Loading SWE-bench dataset...
🔄 Running evaluation pipeline...
✅ PASSED or ❌ FAILED
📊 Accuracy: 1/1 (100.0%)
```

### 3.2 Run Your Target Large Models
```bash
# Test gpt-oss-20b
modal run modal_swebench.py \
  --model="gpt-oss-20b" \
  --instances="django__django-10097" \
  --max-tokens=150

# Test qwen3-coder-30b
modal run modal_swebench.py \
  --model="qwen3-coder-30b-a3b-instruct" \
  --instances="django__django-10097" \
  --max-tokens=150

# Test devstral-small-2507
modal run modal_swebench.py \
  --model="devstral-small-2507" \
  --instances="django__django-10097" \
  --max-tokens=150
```

### 3.3 Run Multiple Instances
```bash
# Multiple instances (batched automatically)
modal run modal_swebench.py \
  --model="gpt-oss-20b" \
  --instances="django__django-10097 requests__requests-863 scikit-learn__scikit-learn-7760" \
  --batch-size=2 \
  --max-tokens=150
```

## Step 4: Using API Models (Anthropic)

### 4.1 Set API Key
```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Option 2: Pass directly
modal run modal_swebench.py \
  --model="claude-3-5-haiku-20241022" \
  --instances="django__django-10097" \
  --api-key="your-api-key-here"
```

### 4.2 Test Anthropic Models
```bash
# Claude models (no GPU needed, much faster)
modal run modal_swebench.py \
  --model="claude-3-5-haiku-20241022" \
  --instances="django__django-10097 requests__requests-863" \
  --max-tokens=200
```

## Step 5: Download Results

### 5.1 Check Modal Dashboard
1. Go to [Modal Dashboard](https://modal.com/apps)
2. Find your `swebench-large-models` app
3. Click on recent runs to see logs
4. Download results from the volumes section

### 5.2 Access Results via CLI
```bash
# List recent results
modal volume ls swebench-results

# Download specific result file
modal volume get swebench-results results_gpt_oss_20b_20240101_120000.json
```

## Step 6: Advanced Usage

### 6.1 Batch Processing Large Lists
```bash
# Process many instances efficiently
modal run modal_swebench.py \
  --model="qwen3-coder-30b-a3b-instruct" \
  --instances="django__django-10097 requests__requests-863 scikit-learn__scikit-learn-7760 flask__flask-4992 matplotlib__matplotlib-18869" \
  --batch-size=3 \
  --max-tokens=200
```

### 6.2 Compare Models
```bash
# Run same instances on different models
models=("gpt-oss-20b" "qwen3-coder-30b-a3b-instruct" "devstral-small-2507")
instances="django__django-10097 requests__requests-863"

for model in "${models[@]}"; do
  echo "Testing $model..."
  modal run modal_swebench.py \
    --model="$model" \
    --instances="$instances" \
    --max-tokens=150
done
```

## Cost Estimates

### GPU Costs (A100 80GB)
| Model Size | Load Time | Inference Time | Total/Instance | 5 Instances |
|------------|-----------|----------------|----------------|-------------|
| gpt-oss-20b | ~3 min | ~2 min | ~$0.40 | ~$2.00 |
| qwen3-coder-30b | ~4 min | ~3 min | ~$0.56 | ~$2.80 |
| devstral-small-2507 | ~2 min | ~1 min | ~$0.24 | ~$1.20 |

### API Model Costs
| Model | Cost/Instance | 5 Instances |
|-------|---------------|-------------|
| claude-3-5-haiku | ~$0.01 | ~$0.05 |
| claude-3-5-sonnet | ~$0.05 | ~$0.25 |

## Troubleshooting

### Model Loading Issues
```bash
# If model fails to load, check Modal logs
modal logs swebench-large-models

# Try with smaller model first
modal run modal_swebench.py --model="gpt2" --instances="django__django-10097"
```

### Docker Issues
```bash
# Check if Docker is working in Modal
modal run modal_swebench.py --model="gpt2" --instances="django__django-10097"
# Look for "🐳 Docker daemon ready" in logs
```

### GPU Memory Issues
```bash
# If OOM, the A100 80GB should handle your models
# Check actual model size requirements
# Consider using model quantization if needed
```

### SWE-bench Evaluation Fails
```bash
# Check SWE-bench tools installation
# Modal automatically installs SWE-bench in the image
# If issues persist, check Modal logs for Docker container errors
```

## Next Steps

1. **Test with your target models** using the commands above
2. **Compare results** across different models
3. **Scale up** to larger instance lists
4. **Analyze confidence patterns** in the downloaded JSON results
5. **Optimize costs** by batching instances efficiently

## Performance Tips

### Optimize for Speed
```bash
# Use batch processing for multiple instances
--batch-size=3

# Use appropriate max-tokens for your use case
--max-tokens=150  # Good balance of quality/speed
```

### Optimize for Cost
```bash
# Start with small model tests
--model="gpt2"

# Use API models for development
--model="claude-3-5-haiku-20241022"

# Batch instances to amortize model loading costs
--batch-size=5
```

Modal gives you the exact same environment as Epoch AI used, with full Docker support and the GPU power needed for your large models!