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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version VARCHAR(50) DEFAULT 'mobilenetv3_v1',
            FOREIGN KEY (card_id) REFERENCES card_templates(id) ON DELETE CASCADE
        );
    """)

    # Create index for faster search (IVFFlat)
    # Note: IVFFlat index should be created AFTER data is loaded for better performance
    # Increase 'lists' parameter as dataset grows (rule of thumb: sqrt(total_rows))
    # For now using 100 lists, increase to 500-1000 when you have 100k+ cards
    try:
        # Check if we have enough data for index (at least 1000 rows recommended)
        cur.execute("SELECT COUNT(*) FROM card_embeddings")
        count = cur.fetchone()[0]

        if count >= 100:  # Minimum threshold for index
            print(f"Creating IVFFlat index for {count} embeddings...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_card_embeddings_ivfflat
                ON card_embeddings
                USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);
            """)
            print("✅ IVFFlat index created successfully")
        else:
            print(f"⚠️ Only {count} embeddings - skipping index creation (need 100+ for optimal performance)")
    except Exception as e:
        print(f"⚠️ Could not create index (this is normal if table is empty): {e}")

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
