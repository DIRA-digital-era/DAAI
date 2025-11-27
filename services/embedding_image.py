# File: DAAI/services/embedding_image.py
import io
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# CPU-only
device = "cpu"

# Lazy-loaded model
_model = None

EMBEDDING_DIM = 1280  # EfficientNet-B1 output

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def _ensure_model_loaded():
    """Load EfficientNet-B1 only on first use (CPU)."""
    global _model
    if _model is None:
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1
        model = efficientnet_b1(weights=weights)
        model.classifier = nn.Identity()
        model.eval()  # CPU eval
        _model = model

def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Convert raw bytes to normalized tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _preprocess(img).unsqueeze(0)

def embed_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Generate L2-normalized image embedding."""
    _ensure_model_loaded()
    x = preprocess_image_bytes(image_bytes)
    with torch.no_grad():
        feats = _model(x)
    arr = feats.numpy()[0].astype(np.float32)
    arr /= np.linalg.norm(arr) + 1e-12
    return arr
