from PIL import Image
import io

def preprocess_image(image_bytes: bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    return img
