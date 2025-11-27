# File: DAAI/services/fallback_store.py
# Enhanced fallback store with integrated diseases.json
# Updated to handle 1792-dim fused embeddings

import threading
import time
import numpy as np
import json
from pathlib import Path

_lock = threading.Lock()

MEM_STORE = {
    "images": {},        # image_id -> {image_id, uploader_id, crop_id, variety_id, file_name, metadata, created_at}
    "embeddings": {},    # image_id -> numpy array
    "inspections": [],   # list of inspections dict
    "users": {},         # user_id -> {id, name, created_at}
    "diseases": {}       # canonical_name -> detailed disease info from diseases.json
}

EXPECTED_EMB_DIM = 1792  # fused image+text embedding dim

# ------------------ Core store functions ------------------

def store_image_embedding(image_id: str, embedding: np.ndarray):
    if embedding.shape[0] != EXPECTED_EMB_DIM:
        print(f"[WARN] Embedding for {image_id} has incorrect dim {embedding.shape[0]}, expected {EXPECTED_EMB_DIM}")
        return
    with _lock:
        MEM_STORE["embeddings"][image_id] = embedding.astype(float).tolist()

def store_image_meta(image_meta: dict):
    with _lock:
        MEM_STORE["images"][image_meta["image_id"]] = image_meta

def register_user_if_missing(user_id: str, name: str = "demo_user"):
    with _lock:
        if user_id not in MEM_STORE["users"]:
            MEM_STORE["users"][user_id] = {
                "id": user_id,
                "name": name,
                "created_at": time.time()
            }

def add_inspection(inspection: dict):
    with _lock:
        MEM_STORE["inspections"].append(inspection)

def get_inspections_for_user(uploader_id: str, limit: int = 50):
    with _lock:
        out = [i for i in MEM_STORE["inspections"] if i.get("uploader_id") == uploader_id]
        out_sorted = sorted(out, key=lambda x: x.get("detection_time", 0), reverse=True)
        return out_sorted[:limit]

def get_user_context(user_id: str):
    with _lock:
        imgs = [m for m in MEM_STORE["images"].values() if m.get("uploader_id") == user_id]
        if not imgs:
            return None
        last = sorted(imgs, key=lambda x: x.get("created_at", 0), reverse=True)[0]
        emb = MEM_STORE["embeddings"].get(last["image_id"])
        return {"last_embedding": emb, "metadata": last, "last_request_time": last.get("created_at")}

# ------------------ Fallback JSON loading ------------------

FALLBACK_JSON_PATH = Path(__file__).parent / "fallback_candidates.json"
DISEASES_JSON_PATH = Path(__file__).parent / "diseases.json"

def load_fallback_json(json_file: str = None):
    """Load fallback candidates JSON into MEM_STORE (images + embeddings)."""
    path = Path(json_file) if json_file else FALLBACK_JSON_PATH
    if not path.exists():
        print(f"[ERROR] Fallback JSON not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    count = 0
    skipped = 0
    for cand_id, cand in candidates.items():
        meta = {
            "image_id": cand_id,
            "file_name": cand.get("file_name"),
            "metadata": cand.get("metadata_json", {}),
            "crop_id": cand.get("crop_label"),
            "variety_id": cand.get("disease_name"),
            "uploader_id": cand.get("uploader_id", "system"),
            "created_at": time.time(),
        }
        store_image_meta(meta)

        embedding_array = np.array(cand.get("embedding", []), dtype=float)
        if embedding_array.shape[0] != EXPECTED_EMB_DIM:
            print(f"[WARN] Skipping {cand_id}, embedding dim {embedding_array.shape[0]} != {EXPECTED_EMB_DIM}")
            skipped += 1
            continue
        store_image_embedding(cand_id, embedding_array)
        count += 1

    print(f"[DONE] Loaded {count} fallback candidates into MEM_STORE. Skipped {skipped} due to dim mismatch.")

def load_diseases_json(json_file: str = None):
    """Load diseases.json into MEM_STORE['diseases'] for full disease metadata access."""
    path = Path(json_file) if json_file else DISEASES_JSON_PATH
    if not path.exists():
        print(f"[ERROR] diseases.json not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    with _lock:
        MEM_STORE["diseases"] = diseases
    print(f"[DONE] Loaded {len(diseases)} diseases into MEM_STORE.")

# ------------------ Vectorization ------------------

VECTOR_EMB_MATRIX = None  # 2D NumPy array of all embeddings
VECTOR_IMAGE_IDS = []     # Parallel list of image_ids

def vectorize_memstore():
    """
    Convert MEM_STORE['embeddings'] to a 2D NumPy matrix for fast similarity search.
    Also keeps a list of image_ids to map back.
    Call after load_fallback_json() and load_diseases_json().
    """
    global VECTOR_EMB_MATRIX, VECTOR_IMAGE_IDS

    embeddings = []
    image_ids = []

    for image_id, emb in MEM_STORE["embeddings"].items():
        try:
            emb_arr = np.array(emb, dtype=np.float32)
            if emb_arr.shape[0] != EXPECTED_EMB_DIM:
                print(f"[WARN] Skipping {image_id}, embedding dim {emb_arr.shape[0]} != {EXPECTED_EMB_DIM}")
                continue
            embeddings.append(emb_arr)
            image_ids.append(image_id)
        except Exception as e:
            print(f"[ERROR] Failed to convert embedding for {image_id}: {e}")
            continue

    if embeddings:
        VECTOR_EMB_MATRIX = np.vstack(embeddings)
        VECTOR_IMAGE_IDS = image_ids
        print(f"[DONE] Vectorized MEM_STORE embeddings: {VECTOR_EMB_MATRIX.shape[0]} images, dim={VECTOR_EMB_MATRIX.shape[1]}")
    else:
        VECTOR_EMB_MATRIX = None
        VECTOR_IMAGE_IDS = []
        print("[WARN] No embeddings to vectorize in MEM_STORE.")
