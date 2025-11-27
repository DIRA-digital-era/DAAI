# File: DAAI/api/upload.py
# Receives image upload + metadata, generates embedding, runs inspection pipeline.

import time
import json
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket
from services import fallback_store
from services.embedding_generator import embed_full, embed_image
from services.inspection_output import inspect_and_produce
import numpy as np
from api.input import safe_metadata

# Optional DB
try:
    from services.db_utils import get_conn_cursor
    USE_DB = True
except Exception:
    USE_DB = False

router = APIRouter()

@router.post("/file")
async def upload_file(
    file: UploadFile = File(...),
    crop_id: Optional[str] = Form(None),
    variety_id: Optional[str] = Form(None),
    uploader_id: str = Form(...),
    notes: Optional[str] = Form(None),
    locale: str = Form("en")
):
    """
    Upload an image, generate embedding, and run inspection.
    Returns full result at the end.
    """
    start_all = time.time()
    image_id = str(uuid.uuid4())
    created_at_ts = time.time()

    try:
        image_bytes = await file.read()

        # Sanitize input metadata
        incoming_meta = {
            "file_name": file.filename,
            "crop_id": crop_id,
            "variety_id": variety_id,
            "uploader_id": uploader_id,
            "notes": notes,
            "locale": locale
        }
        sanitized = safe_metadata(incoming_meta)

        # Compute embeddings
        try:
            fused_embedding = embed_full(image_bytes, sanitized.notes or "")
        except Exception as e_emb:
            try:
                fused_embedding = embed_image(image_bytes)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Embedding failed: {e_emb}; fallback: {e2}")

        # Build observation dict
        new_obs = {
            "_embedding": fused_embedding,
            "image_id": image_id,
            "file_name": sanitized.file_name,
            "uploader_id": sanitized.uploader_id,
            "crop_id": sanitized.crop_id,
            "variety_id": sanitized.variety_id,
            "notes": sanitized.notes,
            "locale": sanitized.locale,
            "created_at": created_at_ts
        }

        # --- Mimic reasoning stream ---
        debug_chunks = []
        # Step 1: Retrieval
        debug_chunks.append({"stage": "retrieval_started"})
        # Simulate retrieval delay
        time.sleep(0.3)
        retrieved = []  # Could call rag_lookup if needed
        debug_chunks.append({"stage": "retrieved", "data": retrieved})

        # Step 2: Reranking
        debug_chunks.append({"stage": "rerank_started"})
        time.sleep(0.2)
        reranked = retrieved  # placeholder
        debug_chunks.append({"stage": "reranked", "data": reranked})

        # Step 3: Generator reasoning
        debug_chunks.append({"stage": "generator_started"})
        time.sleep(0.4)
        # Simulate intermediate reasoning steps
        for step in ["Analyzing leaf texture", "Comparing color patterns", "Matching disease symptoms"]:
            debug_chunks.append({"stage": "generator_chunk", "message": step})
            time.sleep(0.1)

        # Run actual inspection pipeline
        try:
            generator_output = inspect_and_produce(new_obs, top_k=3)
            final_json = generator_output.get("result", {})
        except Exception as e_pipeline:
            final_json = {
                "health_status": "suspect",
                "likely_issue": None,
                "confidence": 0.0,
                "recommended_actions": ["Consult expert - analysis failed"],
                "explanation": f"Pipeline error: {str(e_pipeline)}"
            }

        # Store embedding & metadata
        emb_list = fused_embedding.astype(float).tolist()
        fallback_store.register_user_if_missing(sanitized.uploader_id)
        fallback_store.store_image_meta({
            "image_id": image_id,
            "uploader_id": sanitized.uploader_id,
            "crop_id": sanitized.crop_id,
            "variety_id": sanitized.variety_id,
            "file_name": sanitized.file_name,
            "metadata_json": {"locale": sanitized.locale, "notes": sanitized.notes},
            "created_at": created_at_ts
        })
        fallback_store.store_image_embedding(image_id, fused_embedding)
        fallback_store.add_inspection({
            "inspection_id": str(uuid.uuid4()),
            "crop_image_id": image_id,
            "uploader_id": sanitized.uploader_id,
            "file_name": sanitized.file_name,
            "health_status": final_json.get("health_status"),
            "diagnosis": {"issue": final_json.get("likely_issue")},
            "model_version": "flan-t5-base-v1",
            "confidence": final_json.get("confidence"),
            "detection_time": created_at_ts,
            "created_at": created_at_ts
        })

        total_time = time.time() - start_all
        return {
            "status": "success",
            "image_id": image_id,
            "model_result": final_json,
            "debug": debug_chunks,
            "embedding_dimension": len(emb_list),
            "processing_time_s": round(total_time, 3)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/stream/ws/upload")
async def ws_upload_stream(websocket: WebSocket):
    """
    WebSocket endpoint to mimic streaming reasoning for uploads.
    Client sends a JSON with file bytes base64 and metadata.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        file_bytes = bytes(data["file_bytes"])
        meta = data.get("metadata", {})

        await websocket.send_json({"stage": "accepted", "message": "Upload received"})

        # Simulate embedding computation
        await websocket.send_json({"stage": "embedding_started"})
        time.sleep(0.2)
        embedding = embed_image(file_bytes)
        await websocket.send_json({"stage": "embedding_done", "dim": len(embedding)})

        # Simulate reasoning steps
        steps = ["Analyzing texture", "Matching color patterns", "Checking lesions", "Scoring disease likelihood"]
        for s in steps:
            await websocket.send_json({"stage": "reasoning_chunk", "message": s})
            time.sleep(0.1)

        # Final dummy result
        result = {"health_status": "suspect", "likely_issue": None, "confidence": 0.5}
        await websocket.send_json({"stage": "done", "result": result})
        await websocket.close()
    except Exception as e:
        await websocket.send_json({"stage": "error", "detail": str(e)})
        await websocket.close()
