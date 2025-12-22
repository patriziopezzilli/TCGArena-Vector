import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "tcgarena_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password") # Default placeholder
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    # Enable vector extension support for this connection
    register_vector(conn)
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Ensure extension is enabled
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create table linking card_template to embeddings
    # Using 1280 dimensions for MobileNetV3-Large
    cur.execute("""
        CREATE TABLE IF NOT EXISTS card_embeddings (
            card_id BIGINT PRIMARY KEY,
            embedding vector(1280),
            FOREIGN KEY (card_id) REFERENCES card_templates(id) ON DELETE CASCADE
        );
    """)
    
    # Create index for faster search (IVFFlat) - Optional, requires some data first normally
    # cur.execute("CREATE INDEX IF NOT EXISTS idx_card_embeddings ON card_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);")
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
