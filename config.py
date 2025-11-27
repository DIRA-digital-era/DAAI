import os
import psycopg2
from pgvector.psycopg2 import register_vector
from redis import Redis
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# --------------------------
# PostgreSQL Configuration
# --------------------------
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", 5432))

def get_db_connection():
    """Connect to PostgreSQL. Return (conn, cursor) or (None, None)."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        register_vector(conn)
        cursor = conn.cursor()
        print("[DB INFO] PostgreSQL connection established")
        return conn, cursor
    except Exception as e:
        print(f"[DB ERROR] Could not connect: {e}")
        return None, None

conn, cursor = get_db_connection()

# --------------------------
# Redis Configuration
# --------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

try:
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_client.ping()
    print("[REDIS INFO] Redis connection established")
except Exception as e:
    print(f"[REDIS ERROR] Could not connect to Redis: {e}")
    redis_client = None

# --------------------------
# Model Configuration
# --------------------------
EMBEDDING_DIM = 1536
# Using Flan-T5-base instead of heavy Mistral model
TRANSFORMERS_MODEL = os.getenv("TRANSFORMERS_MODEL", "google/flan-t5-small")

print(f"[MODEL INFO] Using transformer model: {TRANSFORMERS_MODEL}")