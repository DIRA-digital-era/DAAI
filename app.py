from fastapi import FastAPI
from importlib import import_module
from services.model_loader import load_model_bg
import threading

app = FastAPI(title="Crop Inspection Module")


# --------------------------
# Safe router registration
# --------------------------
routers = [
    ("api.upload", "/upload", "Upload"),
    ("api.inspections", "/inspections", "Inspections"),
    ("api.users", "/users", "Users"),
    ("api.stream", "/stream", "Stream"),
   # ("stream.router", "/ws"),
]

for module_name, prefix, tag in routers:
    try:
        module = import_module(module_name)
        if hasattr(module, "router"):
            app.include_router(module.router, prefix=prefix, tags=[tag])
            print(f"[ROUTER INFO] Registered router: {module_name}")
        else:
            print(f"[ROUTER WARNING] {module_name} has no 'router' attribute, skipping.")
    except ModuleNotFoundError as e:
        print(f"[ROUTER ERROR] Could not import {module_name}: {e}")

@app.on_event("startup")
async def startup_event():
    print("[MODEL INIT] Background model load starting...")
    threading.Thread(target=load_model_bg, daemon=True).start()
# --------------------------
# Root endpoint
# --------------------------

@app.get("/")
async def root():
    return {"message": "Crop Inspection API is running"}

# --------------------------
# Health endpoint
# --------------------------
@app.get("/health")
async def health():
    from config import conn, redis_client
    return {
        "db_connected": conn is not None,
        "redis_connected": redis_client is not None,
        "status": "ok" if conn and redis_client else "partial"
    }
