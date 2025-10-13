# OpenRouter Setup Guide

This guide explains how to use OpenRouter with the SWE-bench pipeline to access multiple AI models through a unified API.

## What is OpenRouter?

OpenRouter is a unified API that provides access to multiple AI models (Claude, GPT-4, Gemini, etc.) through a single interface. Benefits include:

- **Single API key** for multiple providers
- **Pay-as-you-go** pricing with transparent costs
- **No subscriptions** required
- **Model comparison** made easy

## Prerequisites

- OpenRouter account (free signup)
- OpenRouter API key
- Python environment with pipeline installed

## Step 1: Get OpenRouter API Key

### 1.1 Sign Up

Visit [OpenRouter](https://openrouter.ai/) and create an account.

### 1.2 Generate API Key

1. Go to [Keys](https://openrouter.ai/keys)
2. Click "Create Key"
3. Give it a name (e.g., "SWE-bench Pipeline")
4. Copy your API key (starts with `sk-or-v1-...`)

### 1.3 Add Credits (Optional)

OpenRouter requires credits for API usage:
1. Go to [Credits](https://openrouter.ai/credits)
2. Add credits (minimum $5 recommended for testing)
3. Monitor usage on the dashboard

## Step 2: Configure API Key

### Option A: Environment Variable (Recommended)

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Option B: .env File

Add to your `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Verify Setup

```bash
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='openrouter/anthropic/claude-sonnet-4')
print(f'Provider: {config.provider}')
print(f'Required key: {config.get_required_api_key_name()}')
print(f'Key found: {config.get_api_key() is not None}')
"
```

Expected output:
```
Provider: ModelProvider.OPENROUTER
Required key: OPENROUTER_API_KEY
Key found: True
```

## Step 3: Model Naming Conventions

OpenRouter uses a specific naming format:

```
openrouter/<provider>/<model-name>
```

### Popular Models

| Provider | Model Name | OpenRouter Format |
|----------|------------|-------------------|
| Anthropic | Claude Sonnet 4 | `openrouter/anthropic/claude-sonnet-4` |
| Anthropic | Claude Sonnet 3.5 | `openrouter/anthropic/claude-3.5-sonnet` |
| Anthropic | Claude Opus 3 | `openrouter/anthropic/claude-3-opus` |
| OpenAI | GPT-4 Turbo | `openrouter/openai/gpt-4-turbo` |
| OpenAI | GPT-4o | `openrouter/openai/gpt-4o` |
| Google | Gemini Pro 1.5 | `openrouter/google/gemini-pro-1.5` |
| Meta | Llama 3 70B | `openrouter/meta-llama/llama-3-70b-instruct` |

### Finding Model Names

Browse available models at: https://openrouter.ai/models

Or use the API:
```bash
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | jq '.data[].id'
```

## Step 4: Running the Pipeline

### Basic Usage

```bash
python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances django__django-10914 \
  --stages 1
```

### With Temperature Parameter

```bash
python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances django__django-10914 \
  --temperature 0.3 \
  --stages 1
```

### Multiple Instances

```bash
python run_pipeline.py \
  --model openrouter/openai/gpt-4-turbo \
  --instances django__django-10914 django__django-10097 \
  --stages 1 2 3
```

## Step 5: Cost Tracking

### View Costs in Output

After running the pipeline, check the stage summary:

```bash
cat output/openrouter-anthropic-claude-sonnet-4/*/stage1_summary.json | grep cost
```

### Monitor on OpenRouter Dashboard

1. Visit [Activity](https://openrouter.ai/activity)
2. View detailed cost breakdown per request
3. Track spending over time

### Cost Comparison (Approximate)

| Model | Cost per 1M input tokens | Cost per 1M output tokens |
|-------|-------------------------|---------------------------|
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Sonnet 3.5 | $3.00 | $15.00 |
| GPT-4 Turbo | $10.00 | $30.00 |
| GPT-4o | $5.00 | $15.00 |
| Gemini Pro 1.5 | $1.25 | $5.00 |
| Llama 3 70B | $0.70 | $0.90 |

**Note**: Prices may vary. Check [OpenRouter pricing](https://openrouter.ai/models) for current rates.

## Step 6: Troubleshooting

### Error: "OPENROUTER_API_KEY not set"

**Solution**: Export the API key or add it to `.env`:
```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Error: "Insufficient credits"

**Solution**: Add credits to your OpenRouter account:
1. Go to https://openrouter.ai/credits
2. Add credits (minimum $5)

### Error: "Model not found"

**Solution**: Verify the model name format:
- Correct: `openrouter/anthropic/claude-sonnet-4`
- Incorrect: `claude-sonnet-4` (missing `openrouter/` prefix)

### Slow Performance

OpenRouter may have rate limits. Check:
1. Your account tier at https://openrouter.ai/settings
2. Current rate limits in the error message
3. Consider adding delays between requests

### Provider Detection Issues

If the provider is not detected correctly:
```bash
python -c "
from pipeline_minisweagent_config import detect_provider, ModelProvider
provider = detect_provider('openrouter/anthropic/claude-sonnet-4')
print(f'Detected: {provider}')
assert provider == ModelProvider.OPENROUTER, 'Provider detection failed'
print('✅ Provider detection working')
"
```

## Advanced Usage

### Custom Configuration

For models with special requirements, use a custom config:

```bash
python run_pipeline.py \
  --model openrouter/meta-llama/llama-3-70b-instruct \
  --custom-config configs/llama3_config.yaml \
  --instances django__django-10914 \
  --stages 1
```

### Batch Processing

Process multiple instances efficiently:

```bash
# Create instance list
cat > instances.txt << EOF
django__django-10914
django__django-10097
django__django-11099
EOF

# Run pipeline
python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances $(cat instances.txt) \
  --stages 1 2 3
```

### Comparing Models

Run the same instances with different models:

```bash
# Test with Claude
python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances django__django-10914 \
  --stages 1 2 3

# Test with GPT-4
python run_pipeline.py \
  --model openrouter/openai/gpt-4-turbo \
  --instances django__django-10914 \
  --stages 1 2 3

# Compare results
python -c "
import json
from pathlib import Path

# Load results
claude_results = json.load(open('output/openrouter-anthropic-claude-sonnet-4/latest/stage3_summary.json'))
gpt4_results = json.load(open('output/openrouter-openai-gpt-4-turbo/latest/stage3_summary.json'))

print(f'Claude Sonnet 4: {claude_results[\"resolved\"]} resolved')
print(f'GPT-4 Turbo: {gpt4_results[\"resolved\"]} resolved')
"
```

## Best Practices

### 1. Start Small

Test with a single instance before scaling:
```bash
python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances django__django-10914 \
  --stages 1
```

### 2. Monitor Costs

Check costs after each run to avoid surprises:
```bash
# View total cost
jq '.total_cost' output/*/latest/stage1_summary.json
```

### 3. Use Appropriate Models

- **Complex fixes**: Claude Sonnet 4, GPT-4 Turbo
- **Simple fixes**: Claude Sonnet 3.5, GPT-4o
- **Cost-conscious**: Gemini Pro 1.5, Llama 3 70B

### 4. Set Temperature Wisely

- **Deterministic fixes** (security, critical bugs): `--temperature 0.0`
- **Creative solutions**: `--temperature 0.7`
- **Default**: No temperature flag (model default)

### 5. Save API Keys Securely

Never commit API keys to git:
```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
```

## Examples

### Example 1: Single Instance with Claude

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances django__django-10914 \
  --stages 1 2 3
```

### Example 2: Multiple Instances with GPT-4

```bash
python run_pipeline.py \
  --model openrouter/openai/gpt-4-turbo \
  --instances django__django-10914 django__django-10097 \
  --temperature 0.3 \
  --stages 1 2 3
```

### Example 3: Cost-Effective Testing with Gemini

```bash
python run_pipeline.py \
  --model openrouter/google/gemini-pro-1.5 \
  --instances django__django-10914 \
  --stages 1
```

## Support

### OpenRouter Support

- Documentation: https://openrouter.ai/docs
- Discord: https://discord.gg/openrouter
- Email: help@openrouter.ai

### Pipeline Issues

For pipeline-specific issues:
1. Check `features/multi-model-support/TODOS.md` for known issues
2. Review logs in `output/<model>/latest/`
3. Verify provider detection is working

## Next Steps

- Try different models and compare results
- Experiment with temperature settings
- Scale up to larger instance sets
- Integrate with your evaluation workflow

Happy experimenting!
