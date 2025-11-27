#DAAI/services/init_fallback.py
from services.fallback_store import load_fallback_json, load_diseases_json, vectorize_memstore, MEM_STORE

def ensure_fallback_loaded():
    if not MEM_STORE["images"]:
        load_fallback_json()
        load_diseases_json()
        vectorize_memstore()
