#DAAI/services/embedding_image.py
import io
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load EfficientNet-B1 (ImageNet weights)
weights = EfficientNet_B1_Weights.IMAGENET1K_V1
_model = efficientnet_b1(weights=weights).to(device)
_model.eval()
_model.classifier = nn.Identity()

EMBEDDING_DIM = 1280  # B1 output

# Hard-coded ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _preprocess(img).unsqueeze(0).to(device)

def embed_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    x = preprocess_image_bytes(image_bytes)
    with torch.no_grad():
        feats = _model(x)
    arr = feats.cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(arr) + 1e-12
    return arr / norm