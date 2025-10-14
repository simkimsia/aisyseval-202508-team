# How to Run First 300 SWE-bench Instances

This guide walks you through running your assigned batch of SWE-bench instances from the first 300.

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher
- Docker installed and running
- WSL (if using Windows)
- Access to the Google Drive for uploading results
- Google API key (will be shared via Telegram)

Repository: https://github.com/simkimsia/aisyseval-202508-team/tree/feat/8-multiple-run and the branch is `feat/8-multiple-run`

Main data folder is: https://drive.google.com/drive/folders/18JLCTe-E6zvny5ZwJECLUz_sf0mBNAas?usp=drive_link

Loom walkthrough: https://www.loom.com/share/0dc3684ab2c64c7283ea60430d0be9d3

## Step 1: Clone the Repository

Clone the repository from the `feat/8-multiple-run` branch:

```bash
git clone -b feat/8-multiple-run https://github.com/simkimsia/aisyseval-202508-team.git
cd aisyseval-202508-team
```

## Step 2: Get Your Assigned Instances

1. Open the assignment spreadsheet: [SWE-bench Instance Assignments](https://docs.google.com/spreadsheets/d/1TJ_PqgZo3nonshEb83lxfYecghwVkTw1TW3xi3ui4Do/edit?usp=sharing)
2. Find your assigned instances in the spreadsheet
3. For initial testing, start with 6-10 instances from your assignment

**Example instances** (replace with your actual assigned instances):
```
django__django-10914
django__django-10097
django__django-11099
django__django-11179
django__django-11283
django__django-11422
```

## Step 3: Setup Environment

### 3.1 Start Docker (and WSL if on Windows)

**On macOS/Linux:**
```bash
# Check if Docker is running
docker ps

# If not running, start Docker
# macOS with OrbStack:
orbstack start
# or use Docker Desktop
```

**On Windows:**
```bash
# Start WSL
wsl

# Inside WSL, check Docker
docker ps
```

### 3.2 Create Virtual Environment

```bash
# Create venv using Python 3.11 or higher
python3.11 -m venv venv

# Activate venv
source venv/bin/activate  # On macOS/Linux/WSL
# or
venv\Scripts\activate     # On Windows (outside WSL)
```

### 3.3 Install Dependencies

```bash
pip install -r requirements_minisweagent.txt
```

### 3.4 Configure API Keys

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << EOF
# Google Gemini API key (will be shared via Telegram)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Add other provider keys if needed
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

**Important:** Replace `your_gemini_api_key_here` with the actual API key shared via Telegram.

### 3.5 Install CodeQL (Required for Stage 4)

**On macOS:**
```bash
brew install --cask codeql
codeql pack download codeql/python-queries
```

**On Linux/WSL:**
```bash
sudo mkdir -p /usr/local/share/codeql/
sudo chown $USER /usr/local/share/codeql/
cd ~
wget https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.23.2/codeql-bundle-linux64.tar.gz
tar xf codeql-bundle-linux64.tar.gz -C /usr/local/share
sudo ln -s /usr/local/share/codeql/codeql /usr/local/bin/codeql
codeql pack download codeql/python-queries
```

## Step 4: Run the Pipeline

Make sure to turn on your venv first!

### Option A: Small Test Run (Recommended First)

Start with 6-10 instances to test everything is working:

```bash
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances django__django-10914 django__django-10097 django__django-11099 \
                django__django-11179 django__django-11283 django__django-11422 \
    --num-runs 1
```

**What this does:**
- Uses Google Gemini 2.5 Pro model
- Runs 6 instances (replace with your assigned instances)
- Performs 1 run per instance
- Executes all 6 stages (patch generation, prediction, evaluation, security, consistency, aggregation)

### Option B: Full Batch Run

Once the test run succeeds, run your full assigned batch:

```bash
# Replace with ALL your assigned instances from the spreadsheet
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances instance1 instance2 instance3 ... instanceN \
    --num-runs 1
```

**Tips:**

- You can split large batches into multiple runs to avoid timeout issues
- Monitor the output for any errors
- The pipeline creates a timestamped directory under `output/gemini/gemini-2.5-pro/`

## Step 5: Monitor Progress

The pipeline will output progress information. You'll see:

```
=== Running Stage 1: Generate Patches ===
Processing instance 1/6: django__django-10914
...
=== Stage 1 Complete ===

=== Running Stage 2: Create Predictions ===
...
```

**Expected Duration:**

- Each instance takes approximately 5-10 minutes
- For 5 instances, expect 30-50 minutes total
- For 50 instances, expect 8 hours

## Step 6: Verify Results

After completion, check the output directory:

```bash
# Find your run directory
ls -lt output/gemini/gemini-2.5-pro/

# Check the summary
cat output/gemini/gemini-2.5-pro/<timestamp>/run_summary.json

# View results in CSV format
cat output/gemini/gemini-2.5-pro/<timestamp>/results.csv
```

**Key files to verify:**
- `run_summary.json` - Overall metrics and results
- `results.csv` - CSV export for easy analysis
- `stage{1-6}_summary.json` - Per-stage status
- Each instance folder contains `run_1/patch.diff` - the actual code fix

## Step 7: Upload Results to Google Drive

Once the run completes successfully:

1. **Locate your output folder:**
   ```bash
   cd output/gemini/gemini-2.5-pro/
   ls -lt  # Find the latest timestamp folder
   ```

2. **Zip the output folder (optional, for faster upload):**
   ```bash
   # Replace <timestamp> with your actual timestamp
   zip -r results_<timestamp>.zip <timestamp>/
   ```

3. **Upload to Google Drive:**
   - Navigate to the shared Google Drive folder (link will be provided)
   - Create a subfolder with your name/ID
   - Upload the entire timestamped folder (or the zip file)
   - Include both the raw outputs and the summary files

4. **Verify upload:**
   - Check that `run_summary.json` and `results.csv` are accessible
   - Ensure all instance folders are present

## Troubleshooting

### Issue: "API key not set"

```bash
# Check if .env file exists and has the key
cat .env

# Verify key is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('GEMINI_API_KEY') else 'NOT FOUND')"
```

### Issue: Docker not running

```bash
# Check Docker status
docker ps

# Start Docker (macOS with OrbStack)
orbstack start

# Or start Docker Desktop manually
```

### Issue: CodeQL not found

```bash
# Check if CodeQL is installed
which codeql

# If not found, follow Step 3.5 above
```

### Issue: Pipeline fails midway

If a stage fails, you can continue from where it stopped:

```bash
# Find your run directory
RUN_DIR="output/gemini/gemini-2.5-pro/<timestamp>"

# Resume from the failed stage (e.g., Stage 3)
python pipeline_3_run_evaluation.py $RUN_DIR
python pipeline_4_security_scan.py $RUN_DIR
python pipeline_5_consistency_check.py $RUN_DIR
python pipeline_6_aggregate_results.py $RUN_DIR
```

### Issue: Out of API credits

Monitor your API usage and costs:
- Check the `metadata.json` file in each instance's run folder
- The `run_summary.json` shows total cost
- Contact the team lead if you need additional credits

## Quick Reference Commands

```bash
# 1. Clone repo
git clone -b feat/8-multiple-run https://github.com/simkimsia/aisyseval-202508-team.git
cd aisyseval-202508-team

# 2. Setup environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_minisweagent.txt

# 3. Create .env with API key
echo "GEMINI_API_KEY=your-key-here" > .env

# 4. Start Docker
docker ps

# 5. Run pipeline (replace instances with yours)
python run_pipeline.py \
    --model gemini/gemini-2.5-pro \
    --instances instance1 instance2 instance3 \
    --num-runs 1 \
    --stages all

# 6. Check results
cat output/gemini/gemini-2.5-pro/*/run_summary.json

# 7. Upload to Google Drive
# (Manual upload through browser or use gdrive CLI)
```

## Expected Output Structure

```
output/
   gemini/gemini-2.5-pro/
       20251014_1830/                    # Your run timestamp
           config.json
           run_summary.json              # � Main results file
           results.csv                   # � CSV export
           stage1_summary.json
           stage2_summary.json
           stage3_summary.json
           stage4_summary.json
           stage5_summary.json
           stage6_summary.json
           predictions_all_run1.json
           evaluation_results_run1.json
           django__django-10914/         # Per instance
               run_1/
                   patch.diff            # Generated code fix
                   prediction.json
                   evaluation.json       # Pass/fail result
                   trajectory.json
                   metadata.json         # Cost & timing
                   security_risk_score.json
```

## Support

- For technical issues: Check the troubleshooting section above
- For API key issues: Contact via Telegram
- For instance assignment questions: Refer to the spreadsheet
- For pipeline questions: See [PIPELINE_README.md](PIPELINE_README.md) and [HOW_TO_USE_PIPELINE.md](HOW_TO_USE_PIPELINE.md)

## Additional Resources

- [PIPELINE_README.md](PIPELINE_README.md) - Detailed pipeline documentation
- [HOW_TO_USE_PIPELINE.md](HOW_TO_USE_PIPELINE.md) - Comprehensive usage guide
- [Multi-Model Guide](features/multi-model-support/MULTI_MODEL_GUIDE.md) - Using different AI providers
- [Example Scripts](examples/) - Ready-to-use examples

---

**Remember:**
1. Start with 6-10 instances to test
2. Monitor Docker and ensure it stays running
3. Check the output regularly for errors
4. Upload results to Google Drive when complete
5. Communicate any issues via Telegram
