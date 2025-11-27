# File: DAAI/services/embedding_generator.py
# Wrapper utilities to produce image, text, and fused embeddings.
# Now includes logging hooks for streaming reasoning steps.

import numpy as np
from .embedding_image import embed_image_from_bytes
from .embedding_text import embed_text

IMAGE_EMB_DIM = 1024
TEXT_EMB_DIM = 768
FUSED_EMB_DIM = IMAGE_EMB_DIM + TEXT_EMB_DIM  # 1792

STEP_LOGGER = None  # Optional callable to log intermediate steps

def log_step(step_name: str, info: dict):
    if STEP_LOGGER:
        try:
            STEP_LOGGER({"step": step_name, "info": info})
        except Exception as e:
            print(f"[WARN] Step logger failed: {e}")

def embed_image(image_bytes: bytes) -> np.ndarray:
    vec = embed_image_from_bytes(image_bytes)
    log_step("image_embedding", {"dim": vec.shape[0]})
    return vec

def embed_text_metadata(text: str) -> np.ndarray:
    vec = embed_text(text)
    log_step("text_embedding", {"dim": vec.shape[0], "preview": text[:50]})
    return vec

def embed_full(image_bytes: bytes, metadata_text: str) -> np.ndarray:
    """
    Generate fused embedding (image + text) with normalization.
    Pre-allocates array to reduce temporary memory usage.
    """
    log_step("start_fusion", {"image_bytes_len": len(image_bytes), "metadata_text_len": len(metadata_text)})

    image_vec = embed_image(image_bytes)
    text_vec = embed_text_metadata(metadata_text)

    # Pre-allocate fused array
    fused = np.empty(FUSED_EMB_DIM, dtype=np.float32)
    fused[:image_vec.shape[0]] = image_vec
    fused[IMAGE_EMB_DIM:] = text_vec

    # Normalize
    fused /= np.linalg.norm(fused) + 1e-12

    log_step("fusion_complete", {"final_dim": fused.shape[0]})
    return fused
