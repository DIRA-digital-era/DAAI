# File: DAAI/api/stream.py
# WebSocket streaming endpoint for inspection with incremental heartbeats and latency logging

import json
import base64
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services import fallback_store
from services.inspection_output import inspect_and_produce
from services.rag_handler import rag_lookup
from services.reranker import rerank_candidates
from services.embedding_generator import embed_image
import numpy as np
import logging

from api.input import safe_metadata

log = logging.getLogger(__name__)
router = APIRouter()

HEARTBEAT_INTERVAL = 1.0  # seconds
CHUNK_SIZE = 400  # chars per chunk


@router.websocket("/ws/inspect")
async def websocket_inspect(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_text()
        try:
            payload = json.loads(msg)
        except Exception:
            await ws.send_json({"error": "invalid JSON payload"})
            await ws.close()
            return

        # --- Parse input ---
        image_id = payload.get("image_id")
        emb = None
        new_obs = None

        if image_id:
            meta = fallback_store.MEM_STORE["images"].get(image_id)
            emb_list = fallback_store.MEM_STORE["embeddings"].get(image_id)
            if meta and emb_list is not None:
                emb = np.array(emb_list, dtype=np.float32)
                stored_metadata = meta.get("metadata_json", {}) or {}
                sanitized = safe_metadata({
                    "file_name": meta.get("file_name"),
                    "crop_id": meta.get("crop_id"),
                    "variety_id": meta.get("variety_id"),
                    "uploader_id": meta.get("uploader_id"),
                    "notes": stored_metadata.get("notes"),
                    "locale": stored_metadata.get("locale"),
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
                }
            else:
                await ws.send_json({"error": "image_id not found in fallback store"})
                await ws.close()
                return

        elif payload.get("image_base64"):
            try:
                b = base64.b64decode(payload["image_base64"])
                emb = embed_image(b)
                sanitized = safe_metadata({
                    "file_name": payload.get("file_name"),
                    "crop_id": payload.get("crop_id"),
                    "variety_id": payload.get("variety_id"),
                    "uploader_id": payload.get("uploader_id"),
                    "notes": payload.get("notes"),
                    "locale": payload.get("locale"),
                })
                new_obs = {
                    "_embedding": emb,
                    "file_name": sanitized.file_name,
                    "uploader_id": sanitized.uploader_id,
                    "crop_id": sanitized.crop_id,
                    "variety_id": sanitized.variety_id,
                    "notes": sanitized.notes,
                    "locale": sanitized.locale,
                }
            except Exception as e:
                await ws.send_json({"error": f"embedding failed: {e}"})
                await ws.close()
                return
        else:
            await ws.send_json({"error": "no valid input: provide image_id or image_base64"})
            await ws.close()
            return

        # --- Stage 1: retrieval ---
        await ws.send_json({"stage": "retrieval_started"})
        retrieved = rag_lookup(emb, top_k=3)
        await ws.send_json({"stage": "retrieved", "data": retrieved})

        # --- Stage 2: rerank ---
        await ws.send_json({"stage": "rerank_started"})
        reranked = rerank_candidates(new_obs, retrieved, temperature=0.0)
        await ws.send_json({"stage": "reranked", "data": reranked})

        # --- Stage 3: generator with heartbeat streaming ---
        await ws.send_json({"stage": "generator_started"})
        start_gen = time.time()

        # heartbeat task
        async def heartbeat():
            while True:
                await ws.send_json({"stage": "generator_heartbeat", "data": "thinking..."})
                await asyncio.sleep(HEARTBEAT_INTERVAL)

        hb_task = asyncio.create_task(heartbeat())

        try:
            res = inspect_and_produce(new_obs, top_k=3)
            final = res.get("result", {})
            debug = res.get("debug", {})
            end_gen = time.time()

            # cancel heartbeat
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass

            # chunked streaming of final JSON
            final_text = json.dumps(final, ensure_ascii=False)
            for i in range(0, len(final_text), CHUNK_SIZE):
                chunk = final_text[i:i + CHUNK_SIZE]
                await ws.send_json({"stage": "generator_chunk", "data": chunk})
                await asyncio.sleep(0.02)  # yield to network

            await ws.send_json({"stage": "generator_final", "data": final})

            # debug includes generation latency
            debug["generation_latency_s"] = round(end_gen - start_gen, 3)
            await ws.send_json({"stage": "debug", "data": debug})

        except Exception as e:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            await ws.send_json({"stage": "generator_error", "error": str(e)})

        await ws.close()

    except WebSocketDisconnect:
        log.info("[WS] Client disconnected")
    except Exception as e:
        log.exception("[WS] Unexpected error: %s", e)
        try:
            await ws.send_json({"error": "server error", "detail": str(e)})
            await ws.close()
        except Exception:
            pass
