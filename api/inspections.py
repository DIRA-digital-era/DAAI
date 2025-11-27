# File: DAAI/api/inspections.py
# Returns past inspections for a given user.
# Uses DB when available; otherwise returns from in-memory fallback_store.
# Also exposes an endpoint to re-run inspection pipeline on an existing stored image_id (if requested).

from fastapi import APIRouter, Query, HTTPException, Form, WebSocket
from services import fallback_store
import time
import uuid
import numpy as np
from api.input import safe_metadata
from services.inspection_output import inspect_and_produce
from services.init_fallback import ensure_fallback_loaded

ensure_fallback_loaded()
router = APIRouter()

@router.post("/rerun")
async def rerun_inspection(image_id: str = Form(...)):
    """
    Re-run inspection on an existing image.
    Returns final JSON + debug chunks.
    """
    start = time.time()
    try:
        emb_list = fallback_store.MEM_STORE["embeddings"].get(image_id)
        img_meta = fallback_store.MEM_STORE["images"].get(image_id)
        if not emb_list or not img_meta:
            raise HTTPException(status_code=404, detail="Image/embedding not found in fallback store")

        emb = np.array(emb_list, dtype=np.float32)
        stored_metadata = img_meta.get("metadata_json", {})
        sanitized = safe_metadata({
            "file_name": img_meta.get("file_name"),
            "crop_id": img_meta.get("crop_id"),
            "variety_id": img_meta.get("variety_id"),
            "uploader_id": img_meta.get("uploader_id"),
            "notes": stored_metadata.get("notes"),
            "locale": stored_metadata.get("locale")
        })

        new_obs = {
            "_embedding": emb,
            "image_id": image_id,
            "file_name": sanitized.file_name,
            "uploader_id": sanitized.uploader_id,
            "crop_id": sanitized.crop_id,
            "variety_id": sanitized.variety_id,
            "notes": sanitized.notes,
            "locale": sanitized.locale,
            "created_at": img_meta.get("created_at")
        }

        # Mimic streaming debug steps
        debug_chunks = []
        stages = ["retrieval_started", "retrieved", "rerank_started", "reranked", "generator_started"]
        for s in stages:
            debug_chunks.append({"stage": s})
            time.sleep(0.1)

        reasoning_steps = ["Checking leaf pattern", "Comparing color distribution", "Estimating disease probability"]
        for step in reasoning_steps:
            debug_chunks.append({"stage": "generator_chunk", "message": step})
            time.sleep(0.05)

        # Run actual inspection
        result = inspect_and_produce(new_obs, top_k=3).get("result", {})

        # Store rerun inspection
        fallback_store.add_inspection({
            "inspection_id": str(uuid.uuid4()),
            "crop_image_id": image_id,
            "uploader_id": sanitized.uploader_id,
            "file_name": sanitized.file_name,
            "health_status": result.get("health_status"),
            "diagnosis": {"issue": result.get("likely_issue")},
            "model_version": result.get("model_version", "mistral-run"),
            "confidence": result.get("confidence"),
            "detection_time": time.time(),
            "created_at": time.time()
        })

        elapsed = time.time() - start
        return {"status": "success", "result": result, "debug": debug_chunks, "latency_s": round(elapsed, 3)}

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.websocket("/stream/ws/inspect")
async def ws_inspect_stream(websocket: WebSocket):
    """
    WebSocket for streaming inspection reasoning.
    Client sends JSON {"image_id": ...} or {"file_bytes": ...}.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        image_id = data.get("image_id")
        file_bytes = data.get("file_bytes")

        if image_id:
            emb_list = fallback_store.MEM_STORE["embeddings"].get(image_id)
            img_meta = fallback_store.MEM_STORE["images"].get(image_id)
            if not emb_list or not img_meta:
                await websocket.send_json({"stage": "error", "detail": "Image not found"})
                await websocket.close()
                return
            emb = np.array(emb_list, dtype=np.float32)
        elif file_bytes:
            emb = embed_image(bytes(file_bytes))
            img_meta = {"file_name": "upload_tmp", "metadata_json": {"notes": "", "locale": "en"}}
        else:
            await websocket.send_json({"stage": "error", "detail": "No image data provided"})
            await websocket.close()
            return

        # Build sanitized observation
        sanitized = safe_metadata({
            "file_name": img_meta.get("file_name"),
            "crop_id": img_meta.get("crop_id"),
            "variety_id": img_meta.get("variety_id"),
            "uploader_id": img_meta.get("uploader_id", "ws_user"),
            "notes": img_meta.get("metadata_json", {}).get("notes"),
            "locale": img_meta.get("metadata_json", {}).get("locale")
        })

        new_obs = {
            "_embedding": emb,
            "image_id": image_id or str(uuid.uuid4()),
            "file_name": sanitized.file_name,
            "uploader_id": sanitized.uploader_id,
            "crop_id": sanitized.crop_id,
            "variety_id": sanitized.variety_id,
            "notes": sanitized.notes,
            "locale": sanitized.locale,
            "created_at": time.time()
        }

        # Streaming mimic
        stages = ["retrieval_started", "retrieved", "rerank_started", "reranked", "generator_started"]
        for s in stages:
            await websocket.send_json({"stage": s})
            time.sleep(0.1)

        reasoning_steps = ["Analyzing leaf", "Checking spots", "Matching color pattern"]
        for step in reasoning_steps:
            await websocket.send_json({"stage": "generator_chunk", "message": step})
            time.sleep(0.05)

        # Final result
        result = inspect_and_produce(new_obs, top_k=3).get("result", {})
        await websocket.send_json({"stage": "done", "result": result})
        await websocket.close()

    except Exception as e:
        await websocket.send_json({"stage": "error", "detail": str(e)})
        await websocket.close()
