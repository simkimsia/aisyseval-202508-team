# Plan

The plan agreed as a team: [Proposal](CS612_Group8_Proposal_Draft.pdf)

## Project Summary

We aim to evaluate LLM safety in software engineering by testing how model consistency correlates with code correctness and security vulnerabilities on SWE-bench tasks.

Our hypothesis: Models will show different consistency patterns when generating correct vs. incorrect/insecure code, and we can use these consistency signals to detect potentially problematic outputs before execution.

**Key objectives:**

- Use consistency checks across multiple generations from frontier models (GPT-4, Claude, etc.) on SWE-bench coding tasks
- Analyze consistency patterns for correct vs. incorrect code solutions
- Test if consistency can predict code that introduces security vulnerabilities
- Develop metrics linking consistency degradation to knowledge integrity in software engineering contexts

## Revised Approach (Based on Learnings)

### Key Discovery: mini-swe-agent vs. Custom Pipeline

**What we learned:**
- Custom evaluation pipelines (like `flexible_pipeline_api.py`) had limited success in generating correct patches
- **mini-swe-agent** (68% resolution rate on SWE-bench verified) significantly outperforms custom approaches
- mini-swe-agent's simplicity (100 lines of Python) and focus on bash-based interaction produces better results

**Why mini-swe-agent works better:**
- Leverages the LM's native capabilities without complex tool scaffolding
- Uses simple bash commands via `subprocess.run`
- Linear message history makes debugging easier
- Stateless execution prevents compounding errors
- Built by Princeton & Stanford team behind SWE-bench

### Current Pipeline Implementation

We have successfully built a modular 4-stage pipeline using mini-swe-agent:

1. **Generate Patches** - Use mini-swe-agent to solve SWE-bench instances
2. **Create Predictions** - Convert patches to SWE-bench prediction format
3. **Run Evaluation** - Execute official SWE-bench evaluation harness
4. **Aggregate Results** - Create comprehensive summaries and analysis

See `PIPELINE_README.md` for complete documentation.

## Step-by-Step Implementation Plan

### Phase 1: Validate Pipeline Correctness ✅ CURRENT PRIORITY

**Goal:** Ensure the evaluation pipeline accurately distinguishes between correct and incorrect patches.

**Approach:**
1. Select a SWE-bench instance that **passed** evaluation (from `output/` folder)
2. Manually modify the generated patch to make it **intentionally fail**
3. Re-run pipeline from Stage 3 (evaluation) onward
4. Verify that the modified patch correctly fails evaluation
5. Document the pipeline's ability to detect incorrect solutions

**Why this matters:**
- Confirms evaluation harness is working correctly
- Establishes baseline for "known bad" patches
- Ensures we can trust the pass/fail signals for consistency analysis

### Phase 2: Generate Multiple Solutions per Instance

**Target Models:** Frontier models via APIs (Claude Sonnet 4, GPT-4, etc.)

**Instance Selection:** Focus on security-relevant SWE-bench instances involving:
- Authentication/authorization bugs (Django session handling, password validation)
- Input validation issues (SQL injection, XSS prevention)
- Cryptographic operations (hashing, encryption, key management)
- File/path handling (directory traversal vulnerabilities)

**Execution:**
1. Run mini-swe-agent pipeline on 10-20 security-relevant instances
2. For each instance, generate **multiple solutions** (5-10 samples per instance)
   - Use different temperature settings (0.1, 0.5, 0.9)
   - Use different models (Claude Sonnet 4, GPT-4, Claude Haiku)
   - Use different random seeds
3. Track which solutions pass/fail evaluation
4. Save all generated patches for consistency analysis

### Phase 3: Consistency Analysis

**Data Collection:**
1. For each instance, collect 5-10 different patch attempts
2. Group patches by instance and model
3. Record evaluation results (pass/fail) for each patch

**Consistency Evaluation:**
1. Use `consistency_evaluator` to measure agreement across multiple patches for the same instance
2. Calculate consistency metrics:
   - Agreement percentage across solutions
   - Confidence scores per solution
   - Variance in patch approaches (token-level, semantic-level)

**Correlation Analysis:**
1. Compare consistency patterns between:
   - Instances where all attempts passed (high confidence)
   - Instances where all attempts failed (wrong approach)
   - Instances with mixed results (model uncertainty)
2. For security-relevant code, analyze consistency around critical sections:
   - Authentication checks
   - Input validation
   - Cryptographic operations
   - Permission/authorization logic

### Phase 4: Security Analysis

**Approach:**
1. Manual review of generated patches for security vulnerabilities
2. Automated scanning with tools:
   - `bandit` for Python security issues
   - `semgrep` for security patterns
3. Correlate security issues with:
   - Low consistency scores
   - High variance in solutions
   - Specific code patterns (e.g., password handling, SQL queries)

**Hypothesis Testing:**
- **H1:** Low consistency predicts incorrect/insecure code
- **H2:** High consistency with incorrect solution indicates systematic model blindspot
- **H3:** Consistency degradation around security-critical code predicts vulnerabilities

### Expected Deliverables

1. **Validated Pipeline** - Confirmed working evaluation harness (Phase 1)
2. **Multi-solution Dataset** - 5-10 patches per instance across 10-20 instances
3. **Consistency Metrics** - Agreement patterns for correct vs. incorrect solutions
4. **Security Analysis** - Correlation between consistency and security vulnerabilities
5. **Research Findings** - Evidence for/against using consistency as vulnerability detector

### Technical Implementation Notes

**Pipeline Architecture:**
```
mini-swe-agent → patch.diff → SWE-bench evaluation → pass/fail
                     ↓
              consistency_evaluator (across multiple patches)
                     ↓
              security analysis + metrics
```

**Key Advantages:**
- Proven patch generation (68% resolution rate with mini-swe-agent)
- Official SWE-bench evaluation harness ensures correctness
- Modular pipeline allows easy extension (add consistency check as Stage 5)
- Comprehensive tracking: cost, time, API calls, resolution rates

**Next Immediate Steps:**
1. ✅ Select passing patch from `output/` folder
2. ✅ Manually break the patch
3. ✅ Re-run evaluation to confirm it fails
4. 🔄 Document validation methodology
5. 🔄 Scale to multiple instances with multiple solutions
