# Flexible SWE-bench Confidence Analysis Pipeline

A flexible pipeline for analyzing model confidence patterns vs. code correctness on SWE-bench instances. Supports both HuggingFace transformers (with logits) and Anthropic API models.

## Features

- 🤖 **Multiple Model Support**: HuggingFace transformers OR Anthropic API models
- 📊 **Logits Extraction**: Token-level confidence scores (HuggingFace only)
- 🧪 **SWE-bench Integration**: Automated patch testing and evaluation
- 📈 **Confidence Analysis**: Correlation between confidence and correctness
- 💾 **Flexible Output**: Command line logs + structured JSON results

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_flexible.txt

# For Anthropic API support (optional)
pip install anthropic>=0.21.0
```

## Quick Start

### HuggingFace Models (with logits)

```bash
# Basic usage - GPT-2 model
python flexible_pipeline.py --model "gpt2" --device "cpu"

# Larger coding model
python flexible_pipeline.py --model "microsoft/CodeGPT-small-py"

# Multiple instances with output file
python flexible_pipeline.py \
  --model "gpt2" \
  --instances "django__django-10097" "requests__requests-863" \
  --output "experiment_results.json"
```

### Anthropic Models (API)

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run with Claude
python flexible_pipeline.py --model "claude-3-5-sonnet-20241022"

# Or pass key directly
python flexible_pipeline.py \
  --model "claude-3-5-haiku-20241022" \
  --api-key "your-key-here" \
  --output "claude_results.json"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name (HF or Anthropic) | `"microsoft/CodeGPT-small-py"` |
| `--instances` | SWE-bench instance IDs | `["django__django-10097"]` |
| `--output` | JSON output file | None (stdout only) |
| `--device` | Device for HF models | `"auto"` (cuda/cpu) |
| `--max-tokens` | Max tokens to generate | `150` |
| `--api-key` | Anthropic API key | Uses `ANTHROPIC_API_KEY` env |

## Output Files

### 1. Command Line Output

Real-time logging with:

- Model loading status
- Token-by-token generation (HF models)
- SWE-bench evaluation results
- Confidence analysis and insights

### 2. JSON Results File (`--output`)

Structured data for each instance:

```json
{
  "instance_id": "django__django-10097",
  "model_name": "gpt2",
  "generated_text": "...",
  "model_patch": "...",
  "confidences": [0.7207, 0.9997, ...],  // Empty for API models
  "logits": [...],                       // Empty for API models
  "is_correct": false,
  "analysis": {
    "avg_confidence": 0.8425,            // null for API models
    "min_confidence": 0.0001,
    "max_confidence": 0.9997,
    "confidence_std": 0.3421,
    "security_avg_confidence": 0.0,
    "security_tokens_found": 0,
    "confidence_degradation": 0.1234,
    "low_confidence_ratio": 0.02,
    "has_confidence_data": true          // false for API models
  },
  "insight": "⚠️ HIGH confidence + WRONG = Dangerous overconfidence"
}
```

## Supported Models

### HuggingFace (with logits)

- `gpt2`
- `microsoft/CodeGPT-small-py`
- `Salesforce/codegen-350M-mono`
- `bigcode/tiny_starcoder`
- Any causal LM on HuggingFace

### Anthropic API (no logits)

- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## Research Workflow

### 1. Single Model Test

```bash
python flexible_pipeline.py \
  --model "gpt2" \
  --instances "django__django-10097" \
  --output "single_test.json"
```

### 2. Multi-Instance Analysis

```bash
python flexible_pipeline.py \
  --model "gpt2" \
  --instances \
    "django__django-10097" \
    "django__django-11019" \
    "requests__requests-863" \
  --output "multi_instance.json"
```

### 3. Model Comparison

```bash
# HuggingFace model (with confidence)
python flexible_pipeline.py \
  --model "gpt2" \
  --instances "django__django-10097" \
  --output "gpt2_results.json"

# Anthropic model (no confidence)
python flexible_pipeline.py \
  --model "claude-3-5-sonnet-20241022" \
  --instances "django__django-10097" \
  --output "claude_results.json"
```

### 4. Capture Full Logs

```bash
python flexible_pipeline.py \
  --model "gpt2" \
  --instances "django__django-10097" \
  --output "results.json" \
  > full_log.txt 2>&1
```

## SWE-bench Integration

The pipeline automatically:

1. **Creates Patch**: Extracts code patches from generated text
2. **Runs Evaluation**: Uses SWE-bench tools to test patches
3. **Determines Pass/Fail**: Checks for `"resolved": true` in evaluation reports
4. **Falls Back Gracefully**: Simulates evaluation if SWE-bench tools unavailable

### SWE-bench Setup (Optional)

For real evaluation (not simulation):

```bash
# Clone SWE-bench tools
git clone https://github.com/princeton-nlp/SWE-bench tools/SWE-bench
cd tools/SWE-bench
pip install -e .
```

## Confidence Analysis

### For HuggingFace Models (with logits)

- **Average Confidence**: Mean probability across tokens
- **Security Token Analysis**: Confidence for security-related keywords
- **Confidence Degradation**: First half vs. second half confidence
- **Low Confidence Ratio**: Percentage of tokens below 50% confidence

### For API Models

- **Focus on Correctness**: SWE-bench pass/fail without confidence data
- **Patch Quality Analysis**: Text-based evaluation of generated patches

## Research Insights

The pipeline categorizes results into:

| Confidence | Correctness | Insight |
|-----------|-------------|---------|
| High (>0.7) | ✅ Correct | "Reliable generation" |
| Low (<0.5) | ✅ Correct | "Lucky guess" |
| High (>0.7) | ❌ Wrong | "**Dangerous overconfidence**" |
| Low (<0.5) | ❌ Wrong | "Uncertain and incorrect" |
| N/A | ✅/❌ | "API model (no confidence)" |

## Troubleshooting

### Common Issues

1. **Model Loading Fails**

   ```bash
   # Try with force download
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2', force_download=True)"
   ```

2. **SWE-bench Tools Missing**
   - Pipeline uses simulation mode automatically
   - Install SWE-bench for real evaluation

3. **Anthropic API Issues**

   ```bash
   # Check API key
   echo $ANTHROPIC_API_KEY

   # Test connection
   pip install anthropic
   python -c "import anthropic; print('Anthropic available')"
   ```

4. **CUDA Issues**

   ```bash
   # Force CPU usage
   python flexible_pipeline.py --device "cpu"
   ```

## Example Research Questions

1. **Do models show different confidence patterns for security vs. non-security code?**
2. **Can low confidence predict vulnerable code generation?**
3. **How do different model sizes affect confidence-correctness correlation?**
4. **Do API models (without confidence) perform better than local models on SWE-bench?**

## File Structure

```
.
├── flexible_pipeline.py          # Main pipeline
├── flexible_pipeline.md          # This documentation
├── requirements_flexible.txt     # Dependencies
├── run_experiments.py           # Example usage scripts
└── tools/
    └── SWE-bench/               # Optional: SWE-bench tools
```

## Contributing

To extend the pipeline:

1. Add new model types in `_is_anthropic_model()`
2. Implement new confidence metrics in `analyze_confidence_patterns()`
3. Extend patch extraction in `extract_code_patch()`
4. Add new insight categories in the results analysis
