# 🤖 AI Evaluator Agent

> **Measure the confidence and consistency of coding models through multi-query evaluation**

The AI Evaluator Agent is designed for **AI safety research, model evaluation, and benchmarking**. It queries coding models multiple times with the same prompt and compares outputs to assess their reliability and consistency.

## 🎯 What It Does

- **Multi-Query Testing**: Runs the same prompt N times against a target model
- **Similarity Analysis**: Compares outputs using AST structure and text similarity
- **Confidence Metrics**: Generates three key metrics to assess model reliability
- **Comprehensive Reporting**: Saves detailed results and summary logs

## ✨ Key Features

### 🔍 Similarity Analysis
- **AST Similarity**: Structure-level comparison (Python only)
- **Text Similarity**: Surface-level comparison
- **Hybrid Similarity**: Weighted combination (70% AST + 30% Text)

### 📊 Confidence Metrics
- **Agreement Percent**: Strict percentage of pairs above threshold
- **Confidence Percent**: Raw average similarity across all pairs
- **Normalized Confidence**: Rescaled values for better interpretability

### 💾 Output Formats
- **Detailed JSON**: Complete results with conversation history
- **Summary CSV**: Quick analysis logs for tracking trends

## 🚀 Quick Start

### 1. Build and Run
```bash
# Build the Docker image
docker build -t evaluator-agent .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v "$(pwd)/results:/app/results" \
  --name evaluator \
  evaluator-agent
```

### 2. Send Evaluation Request
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### 3. Check Results
- **Quick summary**: Displayed in terminal
- **Detailed data**: `results/eval_YYYY-MM-DD_HH-MM-SS.json`
- **Summary log**: `results/summary.csv`

## 📈 How It Works

### Step 1: Pairwise Similarity
For each pair of model outputs:
1. Parse to **AST** (Python only) for structure comparison
2. Compare raw **text** for surface-level similarity
3. Combine into **hybrid similarity**:
   ```
   hybrid = 0.7 × AST + 0.3 × text
   ```

### Step 2: Calculate Metrics

#### Agreement Percent
- Counts pairs with `hybrid ≥ threshold` (default: 0.85)
- Formula: `agreement_% = (consistent_pairs / total_pairs) × 100`
- **Discrete steps**: 20%, 40%, 60%, etc.

#### Confidence Percent
- Average similarity across all pairs
- Formula: `confidence_% = avg(hybrid_i,j) × 100`
- **Continuous values**: e.g., 74.3%

#### Normalized Confidence Percent
- Maps [0.5, 1.0] → [0, 100] for better interpretation
- Formula: `normalized = (avg_hybrid - 0.5) / 0.5 × 100`

## 📋 Example Output

```json
{
  "model": "gpt-4o-mini",
  "question": "Write a Python function that checks if a string is a palindrome.",
  "agreement_percent": 20.0,
  "confidence_percent": 77.3,
  "normalized_confidence_percent": 54.6,
  "n_samples": 5,
  "saved_file": "results/eval_2025-09-18_11-20-03.json"
}
```

## 📂 Results Storage

### Detailed JSON Results
- **Path**: `results/eval_YYYY-MM-DD_HH-MM-SS.json`
- **Contains**: All outputs, pairwise scores, conversation history

### Summary CSV Log
- **Path**: `results/summary.csv`
- **Contains**: One row per evaluation for trend analysis

**Example CSV row:**
```csv
timestamp,model,question,agreement_percent,confidence_percent,normalized_confidence_percent,n_samples,saved_file
2025-09-18_11-20-03,gpt-4o-mini,"Write a Python function that checks if a string is a palindrome.",20.0,77.3,54.6,5,results/eval_2025-09-18_11-20-03.json
```

## 📊 Interpreting Results

| Agreement | Confidence | Interpretation |
|-----------|------------|----------------|
| **High (80-100%)** | **High (>85%)** | ✅ Highly consistent outputs → Strong confidence |
| **Low (<50%)** | **Mid (60-75%)** | ⚠️ Style varies, structure overlaps → Moderate confidence |
| **Any** | **Low (<50%)** | ❌ Significant divergence → Low confidence |

> **⚠️ Important**: Confidence measures **consistency**, not **correctness**. A model can consistently produce the same wrong answer.

## ⚠️ Limitations

- **Language Support**: Only Python benefits from AST-level analysis
- **Correctness vs Consistency**: Metrics measure reliability, not accuracy
- **Score Range**: Raw hybrid scores rarely go below 0.5 due to Python's shared structure