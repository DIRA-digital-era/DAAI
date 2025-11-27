# File: DAAI/services/embedding_generator.py
# Wrapper utilities to produce image, text, and fused embeddings.
# Now includes logging hooks for streaming reasoning steps.

import numpy as np
from .embedding_image import embed_image_from_bytes
from .embedding_text import embed_text

# Default embedding dimensions
IMAGE_EMB_DIM = 1024  # actual output dim of embed_image
TEXT_EMB_DIM = 768    # actual output dim of embed_text
FUSED_EMB_DIM = IMAGE_EMB_DIM + TEXT_EMB_DIM  # 1792

# Optional step logger (callable) to stream intermediate logs
STEP_LOGGER = None  # assign a callable like websocket_send(msg) in runtime

def log_step(step_name: str, info: dict):
    """Log an intermediate reasoning step."""
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
    """Generate fused embedding (image + text) with normalization."""
    log_step("start_fusion", {"image_bytes_len": len(image_bytes), "metadata_text_len": len(metadata_text)})

    image_vec = embed_image(image_bytes)
    text_vec = embed_text_metadata(metadata_text)

    # Ensure concatenated vector has expected 1792 dim
    combined_vec = np.concatenate([image_vec, text_vec])
    if combined_vec.shape[0] != FUSED_EMB_DIM:
        log_step("dimension_mismatch", {
            "expected": FUSED_EMB_DIM,
            "actual": combined_vec.shape[0]
        })
        # Pad or trim to 1792
        if combined_vec.shape[0] < FUSED_EMB_DIM:
            padding = np.zeros(FUSED_EMB_DIM - combined_vec.shape[0], dtype=np.float32)
            combined_vec = np.concatenate([combined_vec, padding])
        else:
            combined_vec = combined_vec[:FUSED_EMB_DIM]

    norm = np.linalg.norm(combined_vec) + 1e-12
    fused_normed = combined_vec / norm

    log_step("fusion_complete", {"final_dim": fused_normed.shape[0], "norm": norm})

    return fused_normed
