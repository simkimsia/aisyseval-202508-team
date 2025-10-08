# Multi-Model Support Guide

This guide covers how to use different AI model providers with the SWE-bench pipeline.

## Overview

The pipeline supports multiple AI providers through automatic provider detection and flexible configuration. You can use:

- **Anthropic** (Claude models) - Direct API
- **OpenAI** (GPT models) - Direct API
- **Google** (Gemini models) - Direct API
- **OpenRouter** - Unified API for all providers
- **Other providers** - Through OpenRouter

## Supported Providers

### Provider Detection

The pipeline automatically detects the provider from the model name:

| Model Name Format | Detected Provider | API Key Required |
|-------------------|-------------------|------------------|
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-*` | OpenAI | `OPENAI_API_KEY` |
| `gemini-*` | Google | `GOOGLE_API_KEY` |
| `openrouter/*` | OpenRouter | `OPENROUTER_API_KEY` |

### Provider Comparison

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **Anthropic** | High quality, good at reasoning | Limited models | Complex bug fixes, security issues |
| **OpenAI** | Fast, widely supported | Can be expensive | General bug fixes, common patterns |
| **Google** | Cost-effective, fast | Newer, less tested | High-volume testing, simple fixes |
| **OpenRouter** | One API for all models | Small markup fee | Model comparison, flexibility |

## Setup by Provider

### Anthropic (Claude)

#### 1. Get API Key

1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Go to [API Keys](https://console.anthropic.com/settings/keys)
3. Create a new key
4. Copy the key (starts with `sk-ant-`)

#### 2. Configure

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Or add to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

#### 3. Available Models

| Model Name | Description | Use Case |
|------------|-------------|----------|
| `claude-sonnet-4-20250514` | Latest Sonnet 4 | Best quality, complex fixes |
| `claude-3-5-sonnet-20241022` | Sonnet 3.5 | Good balance of speed/quality |
| `claude-3-opus-20240229` | Opus 3 | Maximum capability |
| `claude-3-haiku-20240307` | Haiku 3 | Fast, cost-effective |

#### 4. Example Usage

```bash
# Single instance
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 \
  --stages 1 2 3

# With temperature
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 \
  --temperature 0.3 \
  --stages 1

# Multiple instances
python run_pipeline.py \
  --model claude-3-5-sonnet-20241022 \
  --instances django__django-10914 django__django-10097 \
  --stages 1 2 3
```

#### 5. Cost (Approximate)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Sonnet 4 | $3.00 | $15.00 |
| Sonnet 3.5 | $3.00 | $15.00 |
| Opus 3 | $15.00 | $75.00 |
| Haiku 3 | $0.25 | $1.25 |

---

### OpenAI (GPT)

#### 1. Get API Key

1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Go to [API Keys](https://platform.openai.com/api-keys)
3. Create a new secret key
4. Copy the key (starts with `sk-proj-` or `sk-`)

#### 2. Configure

```bash
export OPENAI_API_KEY=sk-proj-your-key-here
```

Or add to `.env`:
```bash
OPENAI_API_KEY=sk-proj-your-key-here
```

#### 3. Available Models

| Model Name | Description | Use Case |
|------------|-------------|----------|
| `gpt-4-turbo` | GPT-4 Turbo | High quality, faster than GPT-4 |
| `gpt-4o` | GPT-4 Optimized | Best balance |
| `gpt-4` | GPT-4 Original | Maximum capability |
| `gpt-3.5-turbo` | GPT-3.5 | Fast, cost-effective |

#### 4. Example Usage

```bash
# Single instance
python run_pipeline.py \
  --model gpt-4-turbo \
  --instances django__django-10914 \
  --stages 1 2 3

# With temperature
python run_pipeline.py \
  --model gpt-4o \
  --instances django__django-10914 \
  --temperature 0.2 \
  --stages 1

# Multiple instances
python run_pipeline.py \
  --model gpt-3.5-turbo \
  --instances django__django-10914 django__django-10097 \
  --stages 1 2 3
```

#### 5. Cost (Approximate)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4 Turbo | $10.00 | $30.00 |
| GPT-4o | $5.00 | $15.00 |
| GPT-4 | $30.00 | $60.00 |
| GPT-3.5 Turbo | $0.50 | $1.50 |

---

### Google (Gemini)

#### 1. Get API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Copy the key

#### 2. Configure

```bash
export GOOGLE_API_KEY=your-google-api-key
```

Or add to `.env`:
```bash
GOOGLE_API_KEY=your-google-api-key
```

#### 3. Available Models

| Model Name | Description | Use Case |
|------------|-------------|----------|
| `gemini-pro` | Gemini Pro | General use |
| `gemini-pro-1.5` | Gemini Pro 1.5 | Latest version |
| `gemini-ultra` | Gemini Ultra | Maximum capability |

#### 4. Example Usage

```bash
# Single instance
python run_pipeline.py \
  --model gemini-pro-1.5 \
  --instances django__django-10914 \
  --stages 1 2 3

# With temperature
python run_pipeline.py \
  --model gemini-pro \
  --instances django__django-10914 \
  --temperature 0.4 \
  --stages 1
```

#### 5. Cost (Approximate)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Gemini Pro 1.5 | $1.25 | $5.00 |
| Gemini Pro | $0.50 | $1.50 |

---

### OpenRouter (Unified API)

OpenRouter provides access to all providers through a single API. See [OPENROUTER_SETUP.md](OPENROUTER_SETUP.md) for detailed setup.

#### 1. Get API Key

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Go to [Keys](https://openrouter.ai/keys)
3. Create a key
4. Copy the key (starts with `sk-or-v1-`)

#### 2. Configure

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Or add to `.env`:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

#### 3. Model Format

```
openrouter/<provider>/<model-name>
```

#### 4. Popular Models

```bash
# Claude through OpenRouter
openrouter/anthropic/claude-sonnet-4
openrouter/anthropic/claude-3.5-sonnet

# GPT through OpenRouter
openrouter/openai/gpt-4-turbo
openrouter/openai/gpt-4o

# Gemini through OpenRouter
openrouter/google/gemini-pro-1.5

# Other models
openrouter/meta-llama/llama-3-70b-instruct
openrouter/mistralai/mistral-large
```

#### 5. Example Usage

```bash
# Claude via OpenRouter
python run_pipeline.py \
  --model openrouter/anthropic/claude-sonnet-4 \
  --instances django__django-10914 \
  --stages 1 2 3

# GPT-4 via OpenRouter
python run_pipeline.py \
  --model openrouter/openai/gpt-4-turbo \
  --instances django__django-10914 \
  --temperature 0.3 \
  --stages 1

# Llama 3 via OpenRouter
python run_pipeline.py \
  --model openrouter/meta-llama/llama-3-70b-instruct \
  --instances django__django-10914 \
  --stages 1
```

#### 6. Benefits

- **One API key** for all models
- **Easy comparison** between models
- **Transparent pricing**
- **No subscriptions** required

---

## Advanced Configuration

### Temperature Parameter

Control randomness/creativity of model output:

```bash
# Deterministic (good for security fixes)
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --temperature 0.0 \
  --instances django__django-10914 \
  --stages 1

# Balanced (default)
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --temperature 0.3 \
  --instances django__django-10914 \
  --stages 1

# Creative (for complex problems)
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --temperature 0.7 \
  --instances django__django-10914 \
  --stages 1
```

**Guidelines:**
- `0.0` - Deterministic, same output every time
- `0.3` - Good default, slight variation
- `0.7` - More creative, different approaches
- `1.0` - Maximum creativity, high variation

### Custom Configuration

For models with special requirements:

```bash
python run_pipeline.py \
  --model some-custom-model \
  --custom-config configs/custom_model.yaml \
  --instances django__django-10914 \
  --stages 1
```

---

## Cost Comparison

### By Task Type

| Task | Recommended Model | Approx. Cost/Instance |
|------|-------------------|----------------------|
| Simple bug fix | GPT-3.5 Turbo, Gemini Pro | $0.01 - $0.05 |
| Medium complexity | Claude Haiku, GPT-4o | $0.05 - $0.20 |
| Complex bug fix | Claude Sonnet, GPT-4 Turbo | $0.20 - $1.00 |
| Security issue | Claude Sonnet 4, GPT-4 | $0.50 - $2.00 |

### Cost Optimization Tips

1. **Start with cheaper models** for testing
2. **Use temperature=0.0** to avoid retries
3. **Test on single instance** before scaling
4. **Monitor costs** in output summaries
5. **Use OpenRouter** for easy price comparison

---

## Model Selection Guide

### When to Use Each Provider

#### Use Anthropic (Claude) When:
- ✅ Complex reasoning required
- ✅ Security-critical fixes
- ✅ Need detailed explanations
- ✅ Long context needed

#### Use OpenAI (GPT) When:
- ✅ Fast iteration needed
- ✅ Well-known patterns
- ✅ General bug fixes
- ✅ Good documentation available

#### Use Google (Gemini) When:
- ✅ Cost is priority
- ✅ High volume testing
- ✅ Simple to medium fixes
- ✅ Fast responses needed

#### Use OpenRouter When:
- ✅ Comparing multiple models
- ✅ Need flexibility
- ✅ Testing different approaches
- ✅ Want unified billing

---

## Common Workflows

### Workflow 1: Model Comparison

Test the same instance with different models:

```bash
# Test with Claude
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 \
  --stages 1 2 3

# Test with GPT-4
python run_pipeline.py \
  --model gpt-4-turbo \
  --instances django__django-10914 \
  --stages 1 2 3

# Test with Gemini
python run_pipeline.py \
  --model gemini-pro-1.5 \
  --instances django__django-10914 \
  --stages 1 2 3

# Compare results
ls -la output/*/latest/stage3_summary.json
```

### Workflow 2: Cost-Effective Testing

Start cheap, scale to quality:

```bash
# Phase 1: Quick test with cheap model
python run_pipeline.py \
  --model gpt-3.5-turbo \
  --instances django__django-10914 \
  --stages 1

# Phase 2: If promising, upgrade to better model
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 \
  --stages 1 2 3
```

### Workflow 3: Batch Processing

Process multiple instances efficiently:

```bash
# Create instance list file
cat > instances.txt << EOF
django__django-10914
django__django-10097
django__django-11099
EOF

# Run with cost-effective model
python run_pipeline.py \
  --model gemini-pro-1.5 \
  --instances $(cat instances.txt) \
  --stages 1 2 3
```

### Workflow 4: Security-Critical Fixes

Use best model with deterministic output:

```bash
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --temperature 0.0 \
  --instances django__django-10097 \
  --stages 1 2 3
```

---

## Troubleshooting

### API Key Issues

**Problem:** "API key not set"

**Solution:**
```bash
# Check which key is needed
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='YOUR_MODEL_NAME')
print(f'Required: {config.get_required_api_key_name()}')
"

# Set the correct key
export ANTHROPIC_API_KEY=your-key  # or OPENAI_API_KEY, etc.
```

### Model Not Found

**Problem:** Model name not recognized

**Solution:**
```bash
# Check provider detection
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='YOUR_MODEL_NAME')
print(f'Detected provider: {config.provider}')
print(f'Model name: {config.model_name}')
"

# Verify model name format
# Anthropic: claude-sonnet-4-20250514
# OpenAI: gpt-4-turbo
# OpenRouter: openrouter/anthropic/claude-sonnet-4
```

### Rate Limits

**Problem:** Too many requests

**Solution:**
- Reduce concurrent instances
- Add delays between requests
- Upgrade API tier
- Use different provider

### High Costs

**Problem:** Unexpected costs

**Solution:**
```bash
# Check costs after each run
jq '.total_cost' output/*/latest/stage1_summary.json

# Use cheaper models for testing
python run_pipeline.py --model gpt-3.5-turbo ...

# Set temperature=0.0 to avoid retries
python run_pipeline.py --temperature 0.0 ...
```

### Provider Detection Issues

**Problem:** Wrong provider detected

**Solution:**
```bash
# Verify provider detection
python -c "
from pipeline_minisweagent_config import detect_provider
provider = detect_provider('YOUR_MODEL_NAME')
print(f'Detected: {provider}')
"

# Update model name format if needed
# Use openrouter/ prefix for OpenRouter models
```

---

## Best Practices

### 1. API Key Security

```bash
# ✅ DO: Use .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo ".env" >> .gitignore

# ❌ DON'T: Hardcode in scripts
# BAD: api_key = "sk-ant-..."
```

### 2. Cost Management

```bash
# Always test with one instance first
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances django__django-10914 \
  --stages 1

# Check cost before scaling
cat output/*/latest/stage1_summary.json | grep total_cost

# Then scale up
python run_pipeline.py \
  --model claude-sonnet-4-20250514 \
  --instances $(cat all_instances.txt) \
  --stages 1 2 3
```

### 3. Model Selection

- **Testing/Development**: Use cheaper models (GPT-3.5, Gemini, Haiku)
- **Production/Critical**: Use best models (Claude Sonnet 4, GPT-4 Turbo)
- **Comparison**: Use OpenRouter for easy switching

### 4. Temperature Settings

- **Security fixes**: `--temperature 0.0`
- **Standard fixes**: `--temperature 0.3`
- **Experimental**: `--temperature 0.7`

### 5. Monitoring

```bash
# Track costs per model
for dir in output/*/latest; do
  model=$(basename $(dirname $dir))
  cost=$(jq -r '.total_cost // 0' $dir/stage1_summary.json 2>/dev/null)
  echo "$model: \$$cost"
done

# Track success rates
for dir in output/*/latest; do
  model=$(basename $(dirname $dir))
  resolved=$(jq -r '.resolved // 0' $dir/stage3_summary.json 2>/dev/null)
  total=$(jq -r '.total_instances // 0' $dir/stage3_summary.json 2>/dev/null)
  echo "$model: $resolved/$total resolved"
done
```

---

## Quick Reference

### Command Template

```bash
python run_pipeline.py \
  --model <MODEL_NAME> \
  --instances <INSTANCE_IDS> \
  [--temperature <0.0-1.0>] \
  [--custom-config <PATH>] \
  --stages <1|2|3|1 2 3>
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...    # For Claude models
OPENAI_API_KEY=sk-proj-...      # For GPT models
GOOGLE_API_KEY=...              # For Gemini models
OPENROUTER_API_KEY=sk-or-v1-... # For OpenRouter
```

### Model Name Formats

```bash
claude-sonnet-4-20250514               # Anthropic direct
gpt-4-turbo                            # OpenAI direct
gemini-pro-1.5                         # Google direct
openrouter/anthropic/claude-sonnet-4   # Via OpenRouter
openrouter/openai/gpt-4-turbo          # Via OpenRouter
```

---

## Support & Resources

### Documentation
- [OpenRouter Setup Guide](OPENROUTER_SETUP.md)
- [Implementation Plan](PLAN.md)
- [TODO Tracking](TODOS.md)

### Provider Documentation
- [Anthropic Claude](https://docs.anthropic.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Google Gemini](https://ai.google.dev/docs)
- [OpenRouter](https://openrouter.ai/docs)

### Pipeline Help

```bash
python run_pipeline.py --help
```

---

## Next Steps

1. **Choose a provider** and get API key
2. **Set environment variable** or add to `.env`
3. **Test with single instance**
4. **Review costs and results**
5. **Scale to more instances**
6. **Compare different models**

Happy bug fixing! 🚀
