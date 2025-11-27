# File: DAAI/services/rag_handler.py
# Robust retrieval logic using pgvector if available, fallback to MEM_STORE if DB fails.
# Uses deterministic SHA256 cache keys and safe DB connections from db_utils.

import json
import hashlib
import numpy as np
from services.db_utils import get_conn_cursor
from services.fallback_store import MEM_STORE, VECTOR_EMB_MATRIX, VECTOR_IMAGE_IDS
from config import redis_client  # existing redis_client

TOP_K = 3
CACHE_TTL = 600  # seconds

def _vec_hash(vec: np.ndarray) -> str:
    return hashlib.sha256(vec.tobytes()).hexdigest()

def _to_list(vec: np.ndarray):
    return vec.astype(float).tolist()

def _py_cosine_sim(a: np.ndarray, b: np.ndarray):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    sim = float(np.dot(a, b))
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim

def rag_lookup(embedding_vector: np.ndarray, top_k: int = TOP_K):
    """
    Returns list of dicts:
    [{"image_id": "...", "distance": <float>, "disease_id": "...", "disease_name": "..."}, ...]
    Uses Redis cache, then DB. Falls back to MEM_STORE only if DB fails.
    """
    if embedding_vector is None:
        return []

    cache_key = "rag:" + _vec_hash(embedding_vector)
    try:
        cached = redis_client.get(cache_key) if redis_client else None
        if cached:
            return json.loads(cached)
    except Exception:
        cached = None

    results = []

    # --- Attempt DB retrieval first ---
    try:
        with get_conn_cursor() as (conn, cur):
            cur.execute("""
                SELECT ci.id::text, ie.embedding <-> %s AS distance, cd.disease_id::text, d.canonical_name
                FROM public.image_embeddings ie
                JOIN public.crop_images ci ON ci.id = ie.image_id
                LEFT JOIN public.crop_diseases cd ON ci.crop_id = cd.crop_id
                LEFT JOIN public.diseases d ON d.id = cd.disease_id
                ORDER BY ie.embedding <-> %s
                LIMIT %s
            """, (_to_list(embedding_vector), _to_list(embedding_vector), top_k))
            rows = cur.fetchall()
            for row in rows:
                image_id, dist, disease_id, disease_name = row
                results.append({
                    "image_id": image_id,
                    "distance": float(dist) if dist is not None else None,
                    "disease_id": disease_id,
                    "disease_name": disease_name
                })
        # DB succeeded
        if results:
            try:
                if redis_client:
                    redis_client.set(cache_key, json.dumps(results, default=str), ex=CACHE_TTL)
            except Exception:
                pass
            return results
    except Exception as db_err:
        print(f"[DB WARNING] DB retrieval failed: {db_err}")

    # --- DB failed, fallback to vectorized MEM_STORE ---
    cand = []
    if VECTOR_EMB_MATRIX is not None and VECTOR_IMAGE_IDS:
        embedding_vector = embedding_vector.astype(np.float32)
        # Check dimension match
        if embedding_vector.shape[0] != VECTOR_EMB_MATRIX.shape[1]:
            print(f"[ERROR] Input embedding dim {embedding_vector.shape[0]} != MEM_STORE dim {VECTOR_EMB_MATRIX.shape[1]}")
            return []

        # Compute cosine distances
        norms = np.linalg.norm(VECTOR_EMB_MATRIX, axis=1) * np.linalg.norm(embedding_vector) + 1e-12
        sims = VECTOR_EMB_MATRIX.dot(embedding_vector) / norms
        dists = 1.0 - np.clip(sims, -1.0, 1.0)
        top_indices = np.argsort(dists)[:top_k]

        for idx in top_indices:
            img_id = VECTOR_IMAGE_IDS[idx]
            dist = float(dists[idx])
            meta = MEM_STORE["images"].get(img_id, {})
            cand.append({
                "image_id": img_id,
                "distance": dist,
                "disease_id": meta.get("disease_id"),
                "disease_name": meta.get("variety_id") or meta.get("disease_name")
            })
    else:
        # fallback to old per-item MEM_STORE loop
        for img_id, emb_list in MEM_STORE.get("embeddings", {}).items():
            try:
                emb = np.array(emb_list, dtype=np.float32)
                if emb.shape[0] != embedding_vector.shape[0]:
                    print(f"[WARN] Skipping {img_id}, shape mismatch: {emb.shape} vs {embedding_vector.shape}")
                    continue
                dist = _py_cosine_sim(embedding_vector, emb)
                meta = MEM_STORE["images"].get(img_id, {})
                cand.append({
                    "image_id": img_id,
                    "distance": float(dist),
                    "disease_id": meta.get("disease_id"),
                    "disease_name": meta.get("variety_id") or meta.get("disease_name")
                })
            except Exception as e:
                print(f"[ERROR] Failed similarity for {img_id}: {e}")
                continue

    results = sorted(cand, key=lambda x: x["distance"])[:top_k]

    # Cache result
    try:
        if redis_client:
            redis_client.set(cache_key, json.dumps(results, default=str), ex=CACHE_TTL)
    except Exception:
        pass

    return results
