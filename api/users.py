# File: DAAI/api/users.py
# User-related endpoints. Returns user list and user context.
# Uses DB when available, otherwise falls back to in-memory store.

from fastapi import APIRouter, Query
from services import fallback_store
import time

# Try DB utils
try:
    from services.db_utils import get_conn_cursor
    USE_DB = True
except Exception:
    USE_DB = False

router = APIRouter()

@router.get("/")
async def list_users():
    """
    List users. If DB unavailable, return in-memory users.
    """
    start = time.time()
    try:
        if USE_DB:
            try:
                with get_conn_cursor() as (conn, cur):
                    cur.execute("SELECT id, coalesce(name,'') FROM public.users LIMIT 100")
                    rows = cur.fetchall()
                users = [{"id": str(r[0]), "name": r[1]} for r in rows]
                return {"users": users, "latency_s": round(time.time() - start, 3)}
            except Exception:
                pass
        # Fallback
        users = []
        for uid, u in fallback_store.MEM_STORE["users"].items():
            users.append({"id": uid, "name": u.get("name")})
        return {"users": users, "latency_s": round(time.time() - start, 3)}
    except Exception as e:
        return {"users": [], "error": str(e)}

@router.get("/context")
async def get_user_context(user_id: str = Query(...)):
    """
    Fetch the last embedding and metadata for a user.
    If DB is unavailable, returns a placeholder response.
    """
    start = time.time()
    try:
        if USE_DB:
            try:
                with get_conn_cursor() as (conn, cur):
                    cur.execute("""
                        SELECT last_embedding, metadata, last_request_time
                        FROM public.user_context
                        WHERE user_id = %s
                        ORDER BY last_request_time DESC
                        LIMIT 1
                    """, (user_id,))
                    row = cur.fetchone()
                if row:
                    return {
                        "user_id": user_id,
                        "last_embedding": row[0],
                        "metadata": row[1],
                        "last_request_time": str(row[2]),
                        "latency_s": round(time.time() - start, 3)
                    }
            except Exception:
                pass

        # Fallback
        ctx = fallback_store.get_user_context(user_id)
        if not ctx:
            return {
                "user_id": user_id,
                "last_embedding": None,
                "metadata": None,
                "last_request_time": None,
                "warning": "Database connection unavailable",
                "latency_s": round(time.time() - start, 3)
            }
        return {
            "user_id": user_id,
            "last_embedding": ctx.get("last_embedding"),
            "metadata": ctx.get("metadata"),
            "last_request_time": ctx.get("last_request_time"),
            "latency_s": round(time.time() - start, 3)
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "last_embedding": None,
            "metadata": None,
            "last_request_time": None,
            "error": str(e)
        }
