# File:conn.py
# Directory: C:\Users\Lenovo\dira_apps\DAAI

import psycopg2
from psycopg2.extras import RealDictCursor

# ---------------- Supabase DB credentials ----------------
DB_HOST = "db.gmzuozfznzxztcwgdwlq.supabase.co"
DB_PORT = 5432
DB_NAME = "gmzuozfznzxztcwgdwlq"       # replace with your DB name
DB_USER = "postgres"                     # replace with your Supabase username
DB_PASSWORD = "F?.FCR?Cn9$&#dY"         # replace with your Supabase password
# ---------------------------------------------------------

try:
    # Connect to Supabase Postgres
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5  # fail fast
    )
    
    # Run a simple query
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT NOW() AS current_time;")
    result = cursor.fetchone()
    print("DB connection successful! Current time:", result['current_time'])
    
    # Clean up
    cursor.close()
    conn.close()
    
except Exception as e:
    print("DB connection failed:", e)
