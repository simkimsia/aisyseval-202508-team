import os
import json
import requests
import difflib
import ast
from itertools import combinations
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ------------------- Similarity Utilities -------------------

def normalize_python_code(code: str) -> str:
    """Normalize Python code into AST dump (ignores formatting & variable names)."""
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        lines = []
        for line in code.splitlines():
            if line.strip().startswith("#"):
                continue
            lines.append(line.strip())
        return "\n".join(lines)

def ast_similarity(a: str, b: str) -> float:
    na = normalize_python_code(a)
    nb = normalize_python_code(b)
    return difflib.SequenceMatcher(None, na, nb).ratio()

def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def hybrid_similarity(a: str, b: str, language: str = "python", alpha: float = 0.7) -> Dict[str, float]:
    if language.lower() == "python":
        ast_sim = ast_similarity(a, b)
    else:
        ast_sim = text_similarity(a, b)
    text_sim = text_similarity(a, b)
    hybrid = alpha * ast_sim + (1 - alpha) * text_sim
    return {"ast": ast_sim, "text": text_sim, "hybrid": hybrid}

def consistency_metrics(outputs: List[str], language: str = "python", threshold: float = 0.85):
    pairs = list(combinations(range(len(outputs)), 2))
    details = []
    hybrids = []
    consistent = 0

    for i, j in pairs:
        sims = hybrid_similarity(outputs[i], outputs[j], language)
        details.append({
            "i": i,
            "j": j,
            "ast_similarity": sims["ast"],
            "text_similarity": sims["text"],
            "hybrid_similarity": sims["hybrid"]
        })
        hybrids.append(sims["hybrid"])
        if sims["hybrid"] >= threshold:
            consistent += 1

    # Agreement % (thresholded)
    agreement_percent = 100.0 * consistent / len(pairs) if pairs else 100.0

    # Raw average hybrid similarity
    avg_hybrid = sum(hybrids) / len(hybrids) if hybrids else 1.0
    confidence_percent = 100.0 * avg_hybrid

    # Normalized confidence (map [0.5, 1.0] → [0, 100])
    if avg_hybrid <= 0.5:
        normalized_confidence = 0.0
    elif avg_hybrid >= 1.0:
        normalized_confidence = 100.0
    else:
        normalized_confidence = ((avg_hybrid - 0.5) / 0.5) * 100.0

    return agreement_percent, confidence_percent, normalized_confidence, details

# ------------------- Request / Response Models -------------------

class SchemaConfig(BaseModel):
    type: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = 0.7

class EvalRequest(BaseModel):
    target_url: str
    prompt: str
    language: str = "python"
    n_samples: int = 5
    threshold: float = 0.85
    request_schema: SchemaConfig = SchemaConfig(type="generic")
    response_schema: SchemaConfig = SchemaConfig(type="generic")

class EvalSummary(BaseModel):
    model: str
    question: str
    agreement_percent: float
    confidence_percent: float
    normalized_confidence_percent: float
    n_samples: int
    saved_file: str

# ------------------- Evaluator Logic -------------------

app = FastAPI(title="Dynamic AI Evaluator Agent", version="5.2")

def call_agent_once(target_url: str, prompt: str, req_schema: SchemaConfig, resp_schema: SchemaConfig) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {}

    if req_schema.type == "generic":
        payload = {"prompt": prompt}

    elif req_schema.type == "openai":
        if not req_schema.api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key missing in request_schema")
        headers["Authorization"] = f"Bearer {req_schema.api_key}"
        payload = {
            "model": req_schema.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant. Return only code."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": req_schema.temperature or 0.7
        }

    elif req_schema.type == "anthropic":
        if not req_schema.api_key:
            raise HTTPException(status_code=400, detail="Anthropic API key missing in request_schema")
        headers["x-api-key"] = req_schema.api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": req_schema.model or "claude-3-sonnet-20240229",
            "max_tokens": 300,
            "temperature": req_schema.temperature or 0.7,
            "messages": [{"role": "user", "content": prompt}]
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported schema type: {req_schema.type}")

    resp = requests.post(target_url, json=payload, headers=headers)
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Target agent error {resp.status_code}: {resp.text}")
    data = resp.json()

    if resp_schema.type == "generic":
        return data.get("answer", str(data))
    elif resp_schema.type == "openai":
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            raise HTTPException(status_code=500, detail=f"Unexpected OpenAI response format: {data}")
    elif resp_schema.type == "anthropic":
        try:
            return data["content"][0]["text"].strip()
        except Exception:
            raise HTTPException(status_code=500, detail=f"Unexpected Anthropic response format: {data}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported response schema type: {resp_schema.type}")

@app.post("/evaluate", response_model=EvalSummary)
def evaluate(req: EvalRequest):
    if req.n_samples < 2:
        raise HTTPException(status_code=400, detail="n_samples must be >= 2")

    outputs = []
    conversation_log = []

    for _ in range(req.n_samples):
        answer = call_agent_once(req.target_url, req.prompt, req.request_schema, req.response_schema)
        outputs.append(answer)
        conversation_log.append({"query": req.prompt, "response": answer})

    # Compute metrics
    agreement_percent, confidence_percent, normalized_confidence, pairwise = consistency_metrics(
        outputs, language=req.language, threshold=req.threshold
    )

    # Build full result
    result = {
        "prompt": req.prompt,
        "agreement_percent": round(agreement_percent, 2),
        "confidence_percent": round(confidence_percent, 2),
        "normalized_confidence_percent": round(normalized_confidence, 2),
        "unique_outputs": len(set(outputs)),
        "example_outputs": [o[:200] for o in outputs],
        "explainability": {
            "outputs": [
                {"index": i, "char_len": len(t), "preview": t[:200] + ("..." if len(t) > 200 else "")}
                for i, t in enumerate(outputs)
            ],
            "pairwise": pairwise,
            "conversation_history": conversation_log
        }
    }

    # Save full result
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"results/eval_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    # ---- NEW PART: append to summary CSV ----
    summary_path = "results/summary.csv"
    header = "timestamp,model,question,agreement_percent,confidence_percent,normalized_confidence_percent,n_samples,saved_file\n"
    row = f"{timestamp},{req.request_schema.model or 'unknown'}," \
          f"\"{req.prompt.replace(',', ';')}\"," \
          f"{round(agreement_percent, 2)}," \
          f"{round(confidence_percent, 2)}," \
          f"{round(normalized_confidence, 2)}," \
          f"{req.n_samples},{filepath}\n"

    # create file with header if missing
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header)
    with open(summary_path, "a") as f:
        f.write(row)

    # Return summary (API response)
    return {
        "model": req.request_schema.model or "unknown",
        "question": req.prompt,
        "agreement_percent": round(agreement_percent, 2),
        "confidence_percent": round(confidence_percent, 2),
        "normalized_confidence_percent": round(normalized_confidence, 2),
        "n_samples": req.n_samples,
        "saved_file": filepath
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
