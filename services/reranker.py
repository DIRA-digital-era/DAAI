# File: DAAI/services/reranker.py
# Reranker implemented with Flan-T5-base using transformers.
# Input: new_observation (dict), candidates (list of dicts from rag_handler)
# Output: list of scored candidates with fields: case_id, score (0..1), reason (str)

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline
from config import TRANSFORMERS_MODEL

log = logging.getLogger(__name__)

# Load model once globally
_reranker_pipe = None

def _ensure_reranker_loaded():
    global _reranker_pipe
    if _reranker_pipe is None:
        try:
            _reranker_pipe = pipeline(
                "text2text-generation",
                model=TRANSFORMERS_MODEL,
                max_length=1024,
                device=-1  # Use CPU (-1), change to 0 for GPU if available
            )
            log.info("[RERANKER] Loaded Flan-T5-base reranker model.")
        except Exception as e:
            log.error(f"[RERANKER] Failed to load model: {e}")
            _reranker_pipe = None
    return _reranker_pipe is not None

# Prompt template optimized for Flan-T5
RERANK_PROMPT_TEMPLATE = """
You are a senior plant pathologist scoring case relevance.

NEW OBSERVATION:
{new_obs_json}

RETRIEVED CASES:
{cases_json}

For each case, provide:
- case_id: string
- relevance_score: number 0.0-1.0
- one_line_reason: short explanation

Return JSON with "cases" list. Only JSON, no other text.
"""

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction helper"""
    # First try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Try to extract JSON object or array
    patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Object pattern
        r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"  # Array pattern
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None

def _format_candidates_for_prompt(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert candidate list to compact JSONable dicts for prompt"""
    out = []
    for c in cands:
        out.append({
            "case_id": c.get("image_id") or c.get("id") or str(c.get("id", "")),
            "distance": c.get("distance"),
            "disease_id": c.get("disease_id"),
            "disease_name": c.get("disease_name"),
            "metadata": c.get("metadata")
        })
    return out

def rerank_candidates(new_obs: Dict[str, Any], candidates: List[Dict[str, Any]], temperature: float = 0.0) -> List[Dict[str, Any]]:
    """
    Rerank candidates using Flan-T5 model.
    Returns list of {"case_id":..., "relevance_score":..., "one_line_reason":...}
    """
    if not candidates:
        return []

    if not _ensure_reranker_loaded():
        log.warning("[RERANK] Model not available, using fallback scoring")
        # Fallback to distance-based scoring
        fallback = []
        for c in candidates:
            dist = c.get("distance")
            score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
            fallback.append({
                "case_id": c.get("image_id"), 
                "relevance_score": score, 
                "one_line_reason": "fallback_distance_score"
            })
        return sorted(fallback, key=lambda x: x["relevance_score"], reverse=True)

    try:
        new_obs_json = json.dumps(new_obs, default=str, ensure_ascii=False, indent=2)
        cases_json = json.dumps(_format_candidates_for_prompt(candidates), default=str, ensure_ascii=False, indent=2)
        
        prompt = RERANK_PROMPT_TEMPLATE.format(
            new_obs_json=new_obs_json,
            cases_json=cases_json
        )

        # Generate with Flan-T5
        result = _reranker_pipe(
            prompt,
            max_length=1024,
            temperature=temperature,
            do_sample=temperature > 0,
            num_return_sequences=1
        )[0]['generated_text']

        parsed = _extract_json_from_text(result)
        
        if parsed and isinstance(parsed, dict) and "cases" in parsed:
            cases = parsed["cases"]
            output = []
            for entry in cases:
                case_id = entry.get("case_id") or entry.get("id")
                score = entry.get("relevance_score") or entry.get("score") or 0.0
                try:
                    score = float(score)
                    score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                except (ValueError, TypeError):
                    score = 0.0
                
                reason = entry.get("one_line_reason") or entry.get("reason") or ""
                
                output.append({
                    "case_id": case_id,
                    "relevance_score": score,
                    "one_line_reason": reason
                })
            
            return sorted(output, key=lambda x: x["relevance_score"], reverse=True)

    except Exception as e:
        log.exception(f"[RERANK] Flan-T5 reranker failed: {e}")

    # Final fallback
    fallback = []
    for c in candidates:
        dist = c.get("distance")
        score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
        fallback.append({
            "case_id": c.get("image_id"),
            "relevance_score": score,
            "one_line_reason": "error_fallback"
        })
    return sorted(fallback, key=lambda x: x["relevance_score"], reverse=True)