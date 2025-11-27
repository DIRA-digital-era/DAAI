# File: DAAI/utils/json_utils.py
# Safe JSON serialization helper used by services.

import json

def safe_serialize(data):
    try:
        return json.dumps(data)
    except Exception:
        # Best-effort fallback
        try:
            return json.dumps(str(data))
        except Exception:
            return json.dumps({"error": "serialization_failed"})
