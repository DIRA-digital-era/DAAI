# File: DAAI/services/embedding_text.py
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"
_tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_MODEL)
_text_model = CLIPTextModel.from_pretrained(CLIP_TEXT_MODEL).to(device)
_text_model.eval()

EMBEDDING_DIM = getattr(_text_model.config, "hidden_size", 512)

def embed_text(text: str) -> np.ndarray:
    inputs = _tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = _text_model(**inputs).last_hidden_state[:,0,:]
    arr = feats.cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(arr) + 1e-12
    return arr / norm
