# File: DAAI/services/inspection_output.py
# Intelligent inspection output with multi-step reasoning and actionable advice.

import json
import time
import logging
from typing import Dict, Any, List, Optional

from services.rag_handler import rag_lookup
from services.reranker import rerank_candidates
from services.fallback_store import MEM_STORE
from transformers import pipeline
from config import TRANSFORMERS_MODEL

log = logging.getLogger(__name__)

_gen_pipe = None
_MODEL_ATTEMPTED = False
_MAX_PROMPT_TOKENS = 1024  # expanded for multi-step reasoning

###############################################
# MODEL LOADING
###############################################
def _ensure_gen_model_loaded():
    global _gen_pipe, _MODEL_ATTEMPTED
    if _gen_pipe is not None:
        return True
    if _MODEL_ATTEMPTED:
        return False
    _MODEL_ATTEMPTED = True
    try:
        _gen_pipe = pipeline(
            "text2text-generation",
            model=TRANSFORMERS_MODEL,
            max_length=_MAX_PROMPT_TOKENS,
            device=-1
        )
        log.info("[INSPECTION_OUTPUT] Loaded Flan-T5-base.")
        return True
    except Exception as e:
        log.error(f"[INSPECTION_OUTPUT] Failed to load Flan-T5: {e}")
        _gen_pipe = None
        return False

###############################################
# SANITIZATION
###############################################
_ILLEGAL_CHARS = ["\u2013", "\u2014", "\u00A0"]

def _sanitize_text(s: str) -> str:
    out = s
    for c in _ILLEGAL_CHARS:
        out = out.replace(c, " ")
    return out

###############################################
# SAFE JSON EXTRACTION
###############################################
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        return None
        return None

###############################################
# SHRINK CASES FOR PROMPT
###############################################
def _shrink_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for c in cases:
        cleaned.append({
            "disease_name": c.get("disease_name"),
            "distance": c.get("distance"),
            "reranker_score": c.get("reranker_score"),
            "metadata": c.get("metadata")
        })
    return cleaned

###############################################
# CONFIDENCE ESTIMATION
###############################################
def _estimate_confidence(best_score: float, avg_score: Optional[float], distance: Optional[float]) -> float:
    base = best_score or 0.0
    avg = avg_score or 0.0
    dist_score = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
    conf = 0.6 * base + 0.25 * avg + 0.15 * dist_score
    return max(0.0, min(1.0, conf))

###############################################
# RERANK MERGE
###############################################
def _combine_scores_and_choose(reranked: List[Dict[str, Any]], candidates: List[Dict[str, Any]], top_n=3):
    merged = []
    for idx, c in enumerate(candidates):
        sc = reranked[idx] if idx < len(reranked) else {}
        merged.append({
            "case_id": c.get("image_id") or c.get("id"),
            "distance": c.get("distance"),
            "disease_id": c.get("disease_id"),
            "disease_name": c.get("disease_name"),
            "metadata": c.get("metadata"),
            "reranker_score": sc.get("relevance_score", 0.0),
            "reranker_reason": sc.get("one_line_reason", "")
        })
    return sorted(merged, key=lambda x: x["reranker_score"], reverse=True)[:top_n]

###############################################
# PROMPT TEMPLATE WITH MULTI-STEP REASONING
###############################################
GEN_PROMPT_TEMPLATE = """
You are an expert plant pathologist AI. Analyze the observation and produce STRICT JSON only.
Follow this multi-step reasoning:

STEP 1: Analyze crop, symptom, and location.
STEP 2: Compare top {top_k} retrieved candidate cases.
STEP 3: Incorporate known disease info (from disease knowledge JSON if available).
STEP 4: Determine health_status and likely issue.
STEP 5: Recommend precise, actionable steps for monitoring, prevention, and treatment.
STEP 6: Highlight critical symptoms and related diseases.

NEW OBSERVATION:
{new_obs}

TOP CANDIDATE CASES:
{cases}

Output JSON STRICTLY with the following schema:
{{
  "health_status": "healthy" | "suspect" | "unhealthy",
  "likely_issue": string or null,
  "confidence": float 0.0-1.0,
  "recommended_actions": [strings],
  "critical_symptoms": [strings],
  "prevention_tips": [strings],
  "treatment_steps": [strings],
  "related_diseases": [strings],
  "explanation": "3-5 sentence reasoning"
}}
"""

###############################################
# MODEL OUTPUT GENERATION
###############################################
def generate_final_json_with_model(new_obs: Dict[str, Any], top_retrieved: List[Dict[str, Any]], reasoning_logs: List[str]) -> Dict[str, Any]:
    reasoning_logs.append("Preparing observation JSON for model input.")
    crop = new_obs.get("crop") or "unknown"
    symptom = new_obs.get("symptom") or new_obs.get("symptoms") or "unspecified"
    location = new_obs.get("location") or "unspecified"

    cases_clean = _shrink_cases(top_retrieved)
    reasoning_logs.append(f"Shrinking {len(top_retrieved)} retrieved cases for prompt.")

    cases_text = json.dumps(cases_clean, ensure_ascii=False)

    prompt = GEN_PROMPT_TEMPLATE.format(
        new_obs=_sanitize_text(json.dumps({
            "crop": crop,
            "symptoms": symptom,
            "location": location
        }, ensure_ascii=False)),
        cases=cases_text,
        top_k=len(cases_clean)
    )
    reasoning_logs.append("Prompt prepared for model generation.")

    try:
        output = _gen_pipe(
            prompt,
            max_length=_MAX_PROMPT_TOKENS,
            temperature=0.0,
            do_sample=False,
            num_return_sequences=1
        )[0]['generated_text']
        reasoning_logs.append("Model output generated successfully.")
    except Exception as e:
        log.error(f"[INSPECTION_GENERATE] Model error: {e}")
        reasoning_logs.append(f"Model generation failed: {e}")
        return None

    parsed = _extract_json_from_text(output)
    if parsed:
        reasoning_logs.append("Parsed JSON output from model successfully.")
        parsed.setdefault("recommended_actions", [])
        parsed.setdefault("critical_symptoms", [])
        parsed.setdefault("prevention_tips", [])
        parsed.setdefault("treatment_steps", [])
        parsed.setdefault("related_diseases", [])
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("explanation", "No explanation generated.")
    else:
        reasoning_logs.append("Model output could not be parsed into JSON.")
    return parsed

###############################################
# FALLBACK LOGIC
###############################################
def generate_final_json_fallback(new_obs: Dict[str, Any], top_retrieved: List[Dict[str, Any]], reasoning_logs: List[str]) -> Dict[str, Any]:
    reasoning_logs.append("Using fallback generation logic.")
    best = top_retrieved[0] if top_retrieved else None
    likely_issue = best.get("disease_name") if best else None
    confidence = best.get("reranker_score", 0.5) if best else 0.5

    reasoning_logs.append(f"Fallback chosen issue: {likely_issue}, confidence: {confidence}")

    return {
        "health_status": "suspect" if confidence < 0.7 else "unhealthy",
        "likely_issue": likely_issue,
        "confidence": confidence,
        "recommended_actions": [
            "Water and fertilize as needed",
            "Monitor for progression of symptoms",
            "Compare plant condition to known disease cases"
        ],
        "critical_symptoms": [best.get("metadata", {}).get("symptoms", "")] if best else [],
        "prevention_tips": [],
        "treatment_steps": [],
        "related_diseases": [],
        "explanation": "Fallback estimation based on similarity to known cases."
    }

###############################################
# MAIN GENERATION
###############################################
def generate_final_json(new_obs: Dict[str, Any], top_retrieved: List[Dict[str, Any]], reasoning_logs: List[str]) -> Dict[str, Any]:
    if _ensure_gen_model_loaded():
        model_out = generate_final_json_with_model(new_obs, top_retrieved, reasoning_logs)
        if model_out:
            return model_out
    return generate_final_json_fallback(new_obs, top_retrieved, reasoning_logs)

###############################################
# ENTRYPOINT
###############################################
def inspect_and_produce(new_obs: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    start = time.time()
    reasoning_logs: List[str] = []
    reasoning_logs.append("Starting inspection process.")

    if new_obs.get("_embedding") is None:
        reasoning_logs.append("Error: no embedding found in new_obs.")
        raise ValueError("inspect_and_produce requires new_obs['_embedding']")

    reasoning_logs.append("Performing RAG lookup.")
    retrieved = rag_lookup(new_obs["_embedding"], top_k=top_k)
    reasoning_logs.append(f"Retrieved {len(retrieved)} candidates from RAG.")

    reasoning_logs.append("Performing reranking of candidates.")
    reranked = rerank_candidates(new_obs, retrieved, temperature=0.0)
    reasoning_logs.append(f"Reranked {len(reranked)} candidates.")

    reasoning_logs.append("Combining scores and selecting top candidates.")
    fused = _combine_scores_and_choose(reranked, retrieved, top_n=top_k)
    reasoning_logs.append(f"Selected {len(fused)} top fused candidates.")

    best_score = fused[0]["reranker_score"] if fused else 0.0
    avg_score = sum(f["reranker_score"] for f in fused) / (len(fused) or 1)
    best_dist = fused[0].get("distance") if fused else None
    confidence = _estimate_confidence(best_score, avg_score, best_dist)
    reasoning_logs.append(f"Estimated confidence: {confidence}")

    final_json = generate_final_json(new_obs, fused, reasoning_logs)
    reasoning_logs.append("Final JSON generated.")

    if "confidence" not in final_json:
        final_json["confidence"] = confidence

    if "health_status" not in final_json:
        final_json["health_status"] = "healthy" if confidence >= 0.7 else "suspect"

    duration = round(time.time() - start, 3)
    reasoning_logs.append(f"Inspection process completed in {duration}s.")

    debug = {
        "retrieved": retrieved,
        "reranked": reranked,
        "fused": fused,
        "confidence_estimate": confidence,
        "duration_s": duration,
        "reasoning_logs": reasoning_logs
    }

    return {"result": final_json, "debug": debug}
