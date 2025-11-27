# File: DAAI/services/embedding_text.py
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

device = "cpu"
CLIP_TEXT_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# Lazy-loaded globals
_tokenizer = None
_text_model = None
EMBEDDING_DIM = 768  # matches original CLIP

def _ensure_model_loaded():
    """Lazy-load CLIP tokenizer and model (CPU)."""
    global _tokenizer, _text_model
    if _tokenizer is None or _text_model is None:
        _tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_MODEL)
        _text_model = CLIPTextModel.from_pretrained(
            CLIP_TEXT_MODEL,
            torch_dtype=torch.float32  # CPU float32
        )
        _text_model.eval()

def embed_text(text: str) -> np.ndarray:
    """Generate L2-normalized text embedding (CPU)."""
    _ensure_model_loaded()
    inputs = _tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        feats = _text_model(**inputs).last_hidden_state[:, 0, :]  # CLS token
    arr = feats.cpu().numpy()[0].astype(np.float32)
    arr /= np.linalg.norm(arr) + 1e-12
    return arr
