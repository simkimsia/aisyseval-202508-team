import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import shared similarity utilities from local module
from .similarity_utils import hybrid_similarity, consistency_metrics

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
