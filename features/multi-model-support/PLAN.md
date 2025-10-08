# Feature Plan: Multi-Model API Support for Pipeline

## Overview

Enable the existing pipeline to work with multiple LLM providers (OpenRouter, OpenAI, Anthropic, etc.) by making the model configuration more flexible and provider-agnostic.

## Current State Analysis

### Existing Pipeline Architecture

- **Stage 1 (`pipeline_1_generate_patches.py`)**: Calls `mini-swe-agent` via subprocess
- **Stage 2-4**: Process outputs from Stage 1 (model-agnostic)
- **Configuration (`pipeline_minisweagent_config.py`)**: Hardcoded `ANTHROPIC_API_KEY` and basic model string

### Current Limitations

1. **Hardcoded API Key Variable**: Only checks `ANTHROPIC_API_KEY` (line 19 in config)
2. **No Provider Detection**: Doesn't distinguish between OpenAI, Anthropic, OpenRouter, etc.
3. **Model Name Pass-through**: Just passes model string to mini-swe-agent without validation
4. **No Temperature/Config Overrides**: Can't set model-specific parameters

### mini-swe-agent Model Support

- Uses **LiteLLM** under the hood → supports 100+ providers
- Accepts model names with provider prefixes (e.g., `anthropic/claude-sonnet-4`, `openai/gpt-4`, `openrouter/anthropic/claude-sonnet-4`)
- Can use environment variables or config files for API keys
- Supports custom configs via YAML for temperature, max tokens, etc.

## Goal

Make the pipeline **provider-agnostic** while maintaining backward compatibility with existing runs.

## Feature Requirements

### 1. Auto-Detect Provider from Model Name

Parse model names to determine which API key is needed:

| Model Pattern | Provider | API Key Env Var | Example |
|--------------|----------|----------------|---------|
| `claude-*`, `anthropic/*` | Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| `gpt-*`, `openai/*` | OpenAI | `OPENAI_API_KEY` | `gpt-4-turbo`, `openai/gpt-4` |
| `openrouter/*` | OpenRouter | `OPENROUTER_API_KEY` | `openrouter/anthropic/claude-sonnet-4` |
| `gemini/*`, `google/*` | Google | `GOOGLE_API_KEY` | `gemini/gemini-2.0-flash` |
| `together/*` | Together AI | `TOGETHER_API_KEY` | `together/meta-llama/Meta-Llama-3.1-70B-Instruct` |
| `deepseek/*` | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-coder` |

### 2. Flexible Model Configuration

Extend `PipelineConfig` to support:

- **Provider auto-detection** from model name
- **Multiple API key environment variables**
- **Optional model parameters** (temperature, top_p, max_tokens)
- **Custom mini-swe-agent config paths** (YAML files)

### 3. API Key Validation

Before running Stage 1:

- Detect provider from model name
- Check if required API key exists
- Provide clear error message with setup instructions

### 4. Support for OpenRouter

OpenRouter requires special handling:

- Uses OpenAI-compatible API
- Model names: `openrouter/provider/model` format
- Single `OPENROUTER_API_KEY` accesses all models

### 5. Cost Tracking by Provider

Update metadata to track:

- Provider used
- Cost calculation method (LiteLLM vs. manual)
- Model-specific pricing (if available)

## Implementation Plan

### Phase 1: Provider Detection & API Key Mapping ✅ HIGH PRIORITY

**Files to modify:**

- `pipeline_minisweagent_config.py`

**Changes:**

1. Add `ModelProvider` enum (Anthropic, OpenAI, OpenRouter, Google, Together, etc.)
2. Add `detect_provider(model_name: str) -> ModelProvider` function
3. Add `get_api_key_env_var(provider: ModelProvider) -> str` function
4. Update `PipelineConfig`:
   - Remove hardcoded `api_key_env_var = "ANTHROPIC_API_KEY"`
   - Add `provider: ModelProvider` (auto-detected in `__post_init__`)
   - Make `get_api_key()` use detected provider

**Testing:**

- Unit tests for provider detection from various model name formats
- Verify backward compatibility with existing `claude-sonnet-4-20250514`

### Phase 2: Enhanced Configuration Options

**Files to modify:**

- `pipeline_minisweagent_config.py`
- `run_pipeline.py`

**Changes:**

1. Add optional model parameters to `PipelineConfig`:

   ```python
   temperature: Optional[float] = None
   top_p: Optional[float] = None
   custom_config_path: Optional[Path] = None
   ```

2. Update `run_pipeline.py` to accept new CLI flags:
   - `--temperature`
   - `--top-p`
   - `--custom-config` (path to mini-swe-agent YAML)
3. Pass these parameters to Stage 1 if provided

**Testing:**

- Test with temperature variations (0.1, 0.5, 0.9)
- Test with custom YAML config for models that don't support temperature

### Phase 3: Update Stage 1 to Use Flexible Config

**Files to modify:**

- `pipeline_1_generate_patches.py`

**Changes:**

1. Use detected provider from config
2. Check correct API key before running mini-swe-agent
3. Add optional flags to mini-swe-agent command:
   - `--temperature` (if supported by model)
   - `--config` (if custom YAML provided)
4. Enhanced error messages:
   - "Missing OPENAI_API_KEY for model gpt-4-turbo"
   - "Set API key: export OPENAI_API_KEY=sk-..."

**Testing:**

- Test with OpenAI model (requires `OPENAI_API_KEY`)
- Test with OpenRouter model (requires `OPENROUTER_API_KEY`)
- Test error handling for missing API keys

### Phase 4: OpenRouter-Specific Support

**Files to modify:**

- `pipeline_minisweagent_config.py`
- Documentation (new `OPENROUTER_SETUP.md`)

**Changes:**

1. Add OpenRouter model name validation
2. Document OpenRouter usage:
   - How to get API key from openrouter.ai
   - Model naming convention: `openrouter/provider/model`
   - Cost tracking considerations
3. Add helper function to list available OpenRouter models

**Testing:**

- Test with various OpenRouter models:
  - `openrouter/anthropic/claude-sonnet-4`
  - `openrouter/openai/gpt-4-turbo`
  - `openrouter/google/gemini-2.0-flash`

### Phase 5: Documentation & Examples

**New files to create:**

1. `MULTI_MODEL_GUIDE.md` - Comprehensive guide for using different providers
2. `examples/openai_example.sh` - Example run with GPT-4
3. `examples/openrouter_example.sh` - Example run with OpenRouter
4. `examples/google_example.sh` - Example run with Gemini

**Documentation content:**

- Setup instructions per provider
- Environment variable reference
- Cost comparison table
- Troubleshooting guide

### Phase 6: Validation & Testing

**Testing matrix:**

| Provider | Model | Expected Behavior |
|----------|-------|------------------|
| Anthropic | `claude-sonnet-4-20250514` | ✅ Current behavior (regression test) |
| OpenAI | `gpt-4-turbo` | ✅ Detects `OPENAI_API_KEY` |
| OpenAI | `openai/gpt-4o` | ✅ Works with provider prefix |
| OpenRouter | `openrouter/anthropic/claude-sonnet-4` | ✅ Uses `OPENROUTER_API_KEY` |
| Google | `gemini/gemini-2.0-flash` | ✅ Uses `GOOGLE_API_KEY` |
| Invalid | No API key set | ❌ Clear error message |

**Validation steps:**

1. Run pipeline with each provider
2. Verify correct API key is checked
3. Verify cost tracking works
4. Verify results are properly aggregated
5. Test with missing API keys (error handling)

## Design Decisions

### Why Not Implement Custom API Clients?

mini-swe-agent already handles all API communication via LiteLLM. Our pipeline just needs to:

1. Detect which API key to check
2. Pass the model name correctly
3. Let mini-swe-agent do the rest

### Why Provider Auto-Detection?

- **User-friendly**: Just specify model name, no extra flags
- **Backward compatible**: Existing runs still work
- **Future-proof**: New providers just need regex pattern

### Why Keep mini-swe-agent as Subprocess?

- **Proven approach**: Current pipeline works (68% resolution rate)
- **Isolation**: mini-swe-agent handles complex agent logic
- **Updates**: We get mini-swe-agent improvements automatically

## Success Criteria

✅ **Backward Compatibility**: Existing runs with Claude still work
✅ **OpenAI Support**: Can run pipeline with `gpt-4-turbo`
✅ **OpenRouter Support**: Can run with OpenRouter models
✅ **Clear Errors**: Helpful messages for missing API keys
✅ **Documentation**: Complete guide for each provider
✅ **Cost Tracking**: Accurate per-provider cost reporting

## Non-Goals (Out of Scope)

❌ **Custom Agent Implementation**: Keep using mini-swe-agent
❌ **API Client Reimplementation**: Let LiteLLM handle it
❌ **Model Registry**: Don't maintain model list (use LiteLLM's)
❌ **Rate Limiting**: Let providers handle it
❌ **Caching**: Keep pipeline stateless for now

## Future Enhancements (After Initial Implementation)

1. **Model Aliases**: Map friendly names to full model strings
   - `claude4` → `claude-sonnet-4-20250514`
   - `gpt4` → `gpt-4-turbo`
2. **Provider Fallback**: Auto-retry with different provider if one fails
3. **Cost Budgets**: Per-provider spending limits
4. **Parallel Multi-Model Runs**: Compare outputs across providers
5. **Model Recommendations**: Suggest best model for instance type

## Migration Path

For existing users:

1. **No action required** for Claude users (backward compatible)
2. **New users**: Just set appropriate API key environment variable
3. **Advanced users**: Can use custom configs and temperature settings

## Questions to Resolve

1. Should we validate model names against LiteLLM registry?
   - **Recommendation**: No, let mini-swe-agent fail with clear error
2. Should we support multiple API keys for same provider?
   - **Recommendation**: Yes, support comma-separated keys for rotation
3. Should we cache model metadata (pricing, context limits)?
   - **Recommendation**: Defer to Phase 2, focus on basic support first

## Implementation Timeline Estimate

- **Phase 1**: 2-3 hours (core provider detection)
- **Phase 2**: 1-2 hours (config enhancements)
- **Phase 3**: 2-3 hours (Stage 1 updates + testing)
- **Phase 4**: 1-2 hours (OpenRouter specifics)
- **Phase 5**: 2-3 hours (documentation)
- **Phase 6**: 2-3 hours (comprehensive testing)

**Total**: ~12-16 hours of development work

## Next Steps

1. Review this plan with team
2. Set up test accounts for OpenRouter, OpenAI (if not already available)
3. Start with Phase 1 (provider detection) - lowest risk, highest value
4. Test with one non-Anthropic model before proceeding
5. Iterate based on findings
