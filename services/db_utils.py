from contextlib import contextmanager
import numpy as np

@contextmanager
def get_conn_cursor():
    """Database connection context manager"""
    try:
        from config import get_db_connection
        conn, cursor = get_db_connection()
        if conn and cursor:
            try:
                yield conn, cursor
            finally:
                cursor.close()
                conn.close()
        else:
            yield None, None
    except Exception as e:
        print(f"[DB_UTILS ERROR] Connection failed: {e}")
        yield None, None