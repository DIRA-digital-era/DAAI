import threading
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import TRANSFORMERS_MODEL

model = None
tokenizer = None
loaded = False

def load_model_bg():
    global model, tokenizer, loaded
    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMERS_MODEL)
        loaded = True
        print(f"[MODEL INIT] {TRANSFORMERS_MODEL} loaded successfully")
    except Exception as e:
        print(f"[MODEL INIT ERROR] {e}")

def ensure_loaded():
    if not loaded:
        raise RuntimeError("Model not loaded yet")
    return model, tokenizer
