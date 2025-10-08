# Multi-Model Support Implementation TODOs

## Guiding Principles

🛡️ **Zero Regression**: Each step must maintain backward compatibility with existing Claude-based runs
✅ **Compile-Safe**: Every commit should be in a working state
🧪 **Test-First**: Validate existing functionality before adding new features

---

## Phase 1: Provider Detection & API Key Mapping

### ✅ Step 1.1: Add Provider Detection (No Breaking Changes)

**Goal**: Add provider detection utilities without modifying existing config behavior

**Files to modify**:

- `pipeline_minisweagent_config.py`

**Changes**:

1. Add `ModelProvider` enum at top of file (new code, no changes to existing)
2. Add `detect_provider(model_name: str) -> ModelProvider` function (new code)
3. Add `get_api_key_env_var(provider: ModelProvider) -> str` function (new code)

**Testing**:

```bash
# Verify existing Claude run still works
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [x] New enum and functions added
- [x] Existing pipeline runs without errors
- [x] No changes to `PipelineConfig` class yet

---

### ✅ Step 1.2: Add Provider Property to Config (Backward Compatible)

**Goal**: Add optional provider detection to `PipelineConfig` without breaking existing usage

**Files to modify**:

- `pipeline_minisweagent_config.py`

**Changes**:

1. Add `provider: Optional[ModelProvider] = None` field to `PipelineConfig`
2. In `__post_init__`, auto-detect provider: `self.provider = detect_provider(self.model_name)`
3. Keep existing `api_key_env_var = "ANTHROPIC_API_KEY"` as fallback
4. Add `provider_name` property that returns string name

**Testing**:

```bash
# Test that provider is detected correctly
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='claude-sonnet-4-20250514')
print(f'Provider: {config.provider}')
print(f'API Key Var: {config.api_key_env_var}')
assert config.get_api_key() is not None, 'API key should be found'
print('✅ Provider detection works')
"

# Verify existing pipeline still works
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] Provider is auto-detected from model name
- [ ] Existing `api_key_env_var` still works
- [ ] Claude pipeline runs without changes

---

### ✅ Step 1.3: Make API Key Dynamic (Backward Compatible)

**Goal**: Update `get_api_key()` to use detected provider, with fallback to hardcoded value

**Files to modify**:

- `pipeline_minisweagent_config.py`

**Changes**:

1. Update `get_api_key()` method:

   ```python
   def get_api_key(self) -> Optional[str]:
       """Get API key from environment, using detected provider."""
       # Try provider-specific key first
       if self.provider:
           env_var = get_api_key_env_var(self.provider)
           api_key = os.getenv(env_var)
           if api_key:
               return api_key

       # Fallback to hardcoded ANTHROPIC_API_KEY for backward compatibility
       return os.getenv(self.api_key_env_var)
   ```

2. Add `get_required_api_key_name()` method that returns expected env var name

**Testing**:

```bash
# Test provider-specific key lookup
python -c "
from pipeline_minisweagent_config import PipelineConfig

# Test Claude (should work with existing ANTHROPIC_API_KEY)
config = PipelineConfig(model_name='claude-sonnet-4-20250514')
print(f'Required key: {config.get_required_api_key_name()}')
assert config.get_api_key() is not None, 'Claude key not found'
print('✅ Claude API key works')

# Test OpenAI (should detect but may not have key - that's OK)
config = PipelineConfig(model_name='gpt-4-turbo')
print(f'Required key: {config.get_required_api_key_name()}')
print(f'Key exists: {config.get_api_key() is not None}')
print('✅ OpenAI provider detection works')
"

# Critical: Verify existing pipeline still works
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] Dynamic API key lookup based on provider
- [ ] Fallback to `ANTHROPIC_API_KEY` still works
- [ ] Claude pipeline runs identically to before

---

### ✅ Step 1.4: Update Stage 1 Error Messages (Backward Compatible)

**Goal**: Improve error messages to show which API key is missing, without changing behavior

**Files to modify**:

- `pipeline_1_generate_patches.py`

**Changes**:

1. Update API key check around line 277:

   ```python
   if not config.get_api_key():
       required_key = config.get_required_api_key_name()
       logger.error(f"❌ {required_key} not set")
       logger.error(f"Required for model: {config.model_name}")
       logger.error("Set it in .env file or environment:")
       logger.error(f"  export {required_key}=your-api-key-here")
       sys.exit(1)
   ```

**Testing**:

```bash
# Test with missing API key (should show helpful error)
unset ANTHROPIC_API_KEY
python pipeline_1_generate_patches.py --model claude-sonnet-4-20250514 --instances django__django-10914 2>&1 | grep "ANTHROPIC_API_KEY"

# Re-export key and test normal run
export ANTHROPIC_API_KEY=your-key-here
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] Error message shows correct API key name
- [ ] Claude pipeline still works normally
- [ ] No functional changes, just better UX

---

### ✅ Step 1.5: Add Provider Info to Metadata (Non-Breaking)

**Goal**: Track provider information in run metadata without affecting execution

**Files to modify**:

- `pipeline_minisweagent_config.py`

**Changes**:

1. Update `to_dict()` method to include provider info:

   ```python
   def to_dict(self) -> dict:
       base_dict = {
           "model_name": self.model_name,
           "provider": self.provider.value if self.provider else "unknown",
           # ... rest of existing fields
       }
       return base_dict
   ```

**Testing**:

```bash
# Run pipeline and check metadata
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1

# Verify provider is in config.json
python -c "
import json
from pathlib import Path
config_path = Path('output/claude-sonnet-4-20250514').glob('*/config.json')
config_path = next(config_path)
config = json.load(open(config_path))
print(f'Provider in metadata: {config.get(\"provider\")}')
assert 'provider' in config, 'Provider should be in metadata'
print('✅ Provider metadata tracking works')
"
```

**Success Criteria**:

- [ ] Provider name appears in config.json
- [ ] Existing pipeline runs without errors
- [ ] Metadata is for tracking only, doesn't affect execution

---

## Phase 2: Enhanced Configuration Options

### ✅ Step 2.1: Add Optional Model Parameters (No CLI Changes Yet)

**Goal**: Add temperature/top_p fields to config without exposing in CLI

**Files to modify**:

- `pipeline_minisweagent_config.py`

**Changes**:

1. Add optional fields to `PipelineConfig`:

   ```python
   # Model parameters (optional)
   temperature: Optional[float] = None
   top_p: Optional[float] = None
   custom_config_path: Optional[Path] = None
   ```

2. Add these to `to_dict()` method
3. No CLI changes yet - just internal support

**Testing**:

```bash
# Test that config accepts new fields
python -c "
from pipeline_minisweagent_config import PipelineConfig

# Test with no parameters (default behavior)
config = PipelineConfig(model_name='claude-sonnet-4-20250514')
print(f'Temperature: {config.temperature}')
print(f'Top-p: {config.top_p}')
print('✅ Optional parameters work')

# Test with parameters
config = PipelineConfig(model_name='claude-sonnet-4-20250514', temperature=0.7)
assert config.temperature == 0.7
print('✅ Temperature parameter works')
"

# Critical: Verify existing pipeline still works
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] Optional fields added to config
- [ ] Default behavior unchanged (None values)
- [ ] Existing pipeline runs without modification

---

### ✅ Step 2.2: Add CLI Flags for Model Parameters

**Goal**: Expose temperature/top_p/config in CLI, but don't use them yet in Stage 1

**Files to modify**:

- `run_pipeline.py`
- `pipeline_minisweagent_config.py` (`create_config_from_args`)

**Changes**:

1. Add CLI flags to `run_pipeline.py`:

   ```python
   parser.add_argument("--temperature", type=float, help="Model temperature")
   parser.add_argument("--top-p", type=float, help="Model top-p sampling")
   parser.add_argument("--custom-config", help="Path to mini-swe-agent YAML config")
   ```

2. Update `create_config_from_args` to pass these through
3. Don't use them in Stage 1 yet - just pass to config

**Testing**:

```bash
# Test CLI accepts new flags
python run_pipeline.py --help | grep temperature
python run_pipeline.py --help | grep top-p
python run_pipeline.py --help | grep custom-config

# Test flags are parsed but don't break anything
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --temperature 0.7 --stages 1

# Critical: Test without flags (default behavior)
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] CLI flags are documented in --help
- [ ] Flags are parsed into config
- [ ] Existing runs without flags work identically

---

## Phase 3: Use Flexible Config in Stage 1

### ✅ Step 3.1: Pass Temperature to mini-swe-agent (Optional)

**Goal**: If temperature is set, pass it to mini-swe-agent command

**Files to modify**:

- `pipeline_1_generate_patches.py`

**Changes**:

1. Update command building around line 66:

   ```python
   cmd = [
       "mini-extra", "swebench-single",
       "--subset", subset,
       "--split", self.config.swebench_split,
       "--model", self.config.model_name,
       "-i", instance_id,
       "-o", str(output_paths["trajectory"]),
   ]

   # Add optional parameters
   if self.config.temperature is not None:
       cmd.extend(["--temperature", str(self.config.temperature)])

   if self.config.exit_immediately:
       cmd.append("--exit-immediately")
   ```

**Testing**:

```bash
# Test with temperature (should pass through)
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --temperature 0.3 --stages 1

# Critical: Test without temperature (default behavior)
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1

# Verify both produce patches
ls output/claude-sonnet-4-20250514/*/django__django-10914/patch.diff
```

**Success Criteria**:

- [ ] Temperature is passed when set
- [ ] Default behavior (no temperature) unchanged
- [ ] Both modes produce valid patches

---

### ✅ Step 3.2: Support Custom Config Path (Optional)

**Goal**: Allow custom YAML config for models with special requirements

**Files to modify**:

- `pipeline_1_generate_patches.py`

**Changes**:

1. Add custom config support to command building:

   ```python
   if self.config.custom_config_path:
       cmd.extend(["--config", str(self.config.custom_config_path)])
   ```

**Testing**:

```bash
# Test without custom config (normal behavior)
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1

# Test with custom config (if you have one)
# python run_pipeline.py --model some-model --custom-config path/to/config.yaml --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] Custom config path is passed when provided
- [ ] Default behavior (no custom config) unchanged
- [ ] Feature is available for future use

---

## Phase 4: OpenRouter-Specific Support

### ✅ Step 4.1: Document OpenRouter Setup

**Goal**: Create comprehensive guide for OpenRouter usage

**Files to create**:

- `features/multi-model-support/OPENROUTER_SETUP.md`

**Content**:

1. How to get OpenRouter API key
2. Model naming conventions
3. Example commands
4. Cost tracking notes

**Testing**:

```bash
# Verify documentation exists and is readable
cat features/multi-model-support/OPENROUTER_SETUP.md
```

**Success Criteria**:

- [ ] Documentation created
- [ ] Clear setup instructions
- [ ] Example commands provided

---

### ✅ Step 4.2: Test OpenRouter Integration (If Key Available)

**Goal**: Validate that OpenRouter models work with the pipeline

**Prerequisites**:

- OpenRouter API key set: `export OPENROUTER_API_KEY=sk-or-...`

**Testing**:

```bash
# Test OpenRouter model detection
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='openrouter/anthropic/claude-sonnet-4')
print(f'Provider: {config.provider}')
print(f'Required key: {config.get_required_api_key_name()}')
assert config.get_required_api_key_name() == 'OPENROUTER_API_KEY'
print('✅ OpenRouter detection works')
"

# If you have OpenRouter key, test actual run
# python run_pipeline.py --model openrouter/anthropic/claude-sonnet-4 --instances django__django-10914 --stages 1
```

**Success Criteria**:

- [ ] OpenRouter provider is detected
- [ ] Correct API key is checked
- [ ] Pipeline runs if key is available

---

## Phase 5: Documentation & Examples

### ✅ Step 5.1: Create Multi-Model Guide

**Goal**: Comprehensive guide for all supported providers

**Files to create**:

- `features/multi-model-support/MULTI_MODEL_GUIDE.md`

**Content**:

1. Overview of supported providers
2. Setup instructions per provider
3. Model naming conventions
4. Cost comparison
5. Troubleshooting tips

**Testing**:

```bash
# Verify guide exists
cat features/multi-model-support/MULTI_MODEL_GUIDE.md
```

**Success Criteria**:

- [ ] Guide covers all major providers
- [ ] Clear examples for each
- [ ] Troubleshooting section included

---

### ✅ Step 5.2: Create Example Scripts

**Goal**: Runnable examples for each provider

**Files to create**:

- `examples/openai_example.sh`
- `examples/openrouter_example.sh`
- `examples/google_example.sh`

**Content**: Each script shows:

1. How to set API key
2. Example run command
3. Expected output location

**Testing**:

```bash
# Verify scripts exist and are executable
chmod +x examples/*.sh
ls -l examples/

# Read through each script
cat examples/openai_example.sh
```

**Success Criteria**:

- [ ] Scripts created and executable
- [ ] Clear comments explaining each step
- [ ] Ready for users to copy/paste

---

## Phase 6: Validation & Testing

### ✅ Step 6.1: Create Test Suite

**Goal**: Automated tests for provider detection

**Files to create**:

- `tests/test_provider_detection.py`

**Content**:

```python
def test_anthropic_detection():
    """Test Claude model detection."""
    assert detect_provider("claude-sonnet-4-20250514") == ModelProvider.ANTHROPIC

def test_openai_detection():
    """Test OpenAI model detection."""
    assert detect_provider("gpt-4-turbo") == ModelProvider.OPENAI

def test_openrouter_detection():
    """Test OpenRouter model detection."""
    assert detect_provider("openrouter/anthropic/claude-sonnet-4") == ModelProvider.OPENROUTER
```

**Testing**:

```bash
# Run tests
python -m pytest tests/test_provider_detection.py -v
```

**Success Criteria**:

- [ ] Tests pass for all providers
- [ ] Edge cases covered
- [ ] Tests can be run in CI

---

### ✅ Step 6.2: Regression Testing

**Goal**: Verify all changes maintain backward compatibility

**Testing Matrix**:

| Test Case | Command | Expected Result |
|-----------|---------|-----------------|
| Original Claude | `python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914` | ✅ Works identically |
| Claude with temp | `python run_pipeline.py --model claude-sonnet-4-20250514 --temperature 0.3 --instances django__django-10914` | ✅ Works with parameter |
| OpenAI detection | Config with `gpt-4-turbo` | ✅ Detects OpenAI provider |
| Missing key | Unset API key | ❌ Clear error message |

**Validation Script**:

```bash
#!/bin/bash
# Run regression tests

echo "🧪 Running regression tests..."

# Test 1: Original Claude behavior
echo "Test 1: Original Claude pipeline"
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1
if [ $? -eq 0 ]; then
    echo "✅ Test 1 passed"
else
    echo "❌ Test 1 FAILED - REGRESSION DETECTED"
    exit 1
fi

# Test 2: Provider detection
echo "Test 2: Provider detection"
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='claude-sonnet-4-20250514')
assert config.provider is not None
print('✅ Test 2 passed')
"

# Test 3: API key lookup
echo "Test 3: API key lookup"
python -c "
from pipeline_minisweagent_config import PipelineConfig
config = PipelineConfig(model_name='claude-sonnet-4-20250514')
assert config.get_api_key() is not None
print('✅ Test 3 passed')
"

echo "🎉 All regression tests passed!"
```

**Success Criteria**:

- [ ] All original functionality works
- [ ] New features are optional
- [ ] No breaking changes introduced

---

## Rollback Plan

If any step introduces regressions:

1. **Immediate**: Revert the specific commit
2. **Diagnose**: Identify what broke backward compatibility
3. **Fix**: Add compatibility layer or modify approach
4. **Re-test**: Run full regression suite before proceeding

**Git Safety**:

```bash
# Before each phase, create a checkpoint
git add -A
git commit -m "checkpoint: Phase X Step Y working"

# If something breaks
git revert HEAD  # or git reset --hard HEAD~1
```

---

## Progress Tracking

- [ ] Phase 1: Provider Detection (5 steps)
- [ ] Phase 2: Configuration Options (2 steps)
- [ ] Phase 3: Stage 1 Integration (2 steps)
- [ ] Phase 4: OpenRouter Support (2 steps)
- [ ] Phase 5: Documentation (2 steps)
- [ ] Phase 6: Testing & Validation (2 steps)

**Total**: 15 steps, all compile-safe and regression-proof

---

## Notes

- **After each step**: Run `python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --stages 1`
- **Before committing**: Verify existing functionality works
- **If stuck**: Revert to last working checkpoint
- **Priority**: Backward compatibility > New features
