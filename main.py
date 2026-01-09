from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from vectorizer import CardVectorizer
from database import get_db_connection, init_db
import os
import glob

app = FastAPI(title="TCGArena Visual Search")
vectorizer = CardVectorizer()

# Configuration
IMAGE_STORAGE_PATH = os.getenv("IMAGE_STORAGE_PATH", "/TCGArena-Images/")

class SearchResult(BaseModel):
    card_id: int
    confidence: float

class ReindexRequest(BaseModel):
    force_reindex: bool = False

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/reindex")
async def reindex_embeddings(request: ReindexRequest, background_tasks: BackgroundTasks):
    """
    Re-index all embeddings with normalized vectors and updated model.
    This is needed after switching to cosine similarity.
    """
    background_tasks.add_task(reindex_all_embeddings, request.force_reindex)
    return {"message": "Re-indexing started in background"}

def reindex_all_embeddings(force: bool = False):
    """
    Re-process all existing embeddings with normalization and update model_version.
    """
    print("Starting re-indexing process...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get all existing embeddings
    cur.execute("SELECT card_id, embedding FROM card_embeddings")
    rows = cur.fetchall()
    
    updated_count = 0
    for card_id, embedding_bytes in rows:
        # Convert bytes back to list (pgvector stores as bytes)
        embedding_list = list(embedding_bytes)
        
        # Normalize the vector
        import numpy as np
        embedding_array = np.array(embedding_list)
        normalized = embedding_array / np.linalg.norm(embedding_array)
        normalized_list = normalized.tolist()
        
        # Update in database
        cur.execute("""
            UPDATE card_embeddings 
            SET embedding = %s::vector, model_version = 'mobilenetv3_v1_normalized'
            WHERE card_id = %s
        """, (normalized_list, card_id))
        
        updated_count += 1
        if updated_count % 100 == 0:
            print(f"Re-indexed {updated_count} embeddings...")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Re-indexing complete: {updated_count} embeddings normalized")
    
    # Re-create index with cosine operator
    init_db()

@app.post("/search", response_model=List[SearchResult])
async def search_card(file: UploadFile = File(...)):
    """
    Searches for the closest matching card based on the uploaded image.
    """
    image_bytes = await file.read()
    query_vector = vectorizer.vectorize_image(image_bytes)
    
    if query_vector is None:
        raise HTTPException(status_code=400, detail="Could not process image")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Perform Nearest Neighbor Search using Cosine Similarity (better for embeddings)
    # The <=> operator is provided by pgvector for Cosine distance (1 - cosine similarity)
    # Cosine similarity is better than L2 for normalized embeddings
    # Increased from LIMIT 5 to LIMIT 20 for better recall
    cur.execute("""
        SELECT card_id, (1 - (embedding <=> %s::vector)) as cosine_similarity
        FROM card_embeddings
        ORDER BY cosine_similarity DESC
        LIMIT 20;
    """, (query_vector,))

    results = cur.fetchall()
    cur.close()
    conn.close()

    # Convert Cosine Similarity to confidence percentage
    # Cosine similarity ranges from -1 (opposite) to 1 (identical)
    # For normalized embeddings, typically 0.0 to 1.0 (1.0 = perfect match)
    def similarity_to_confidence(similarity: float) -> float:
        """
        Converts cosine similarity to confidence percentage (0-100)

        Similarity 1.0  → 100% confidence (perfect match)
        Similarity 0.8  → 80% confidence (good match)
        Similarity 0.5  → 50% confidence (moderate match)
        Similarity 0.0  → 0% confidence (no match)
        """
        # Clamp similarity to 0-1 range (normalized embeddings)
        clamped = max(0.0, min(similarity, 1.0))

        # Convert to percentage
        confidence = clamped * 100.0

        return round(confidence, 2)

    return [
        {
            "card_id": row[0],
            "confidence": similarity_to_confidence(float(row[1]))
        }
        for row in results
    ]

def process_batch_images():
    """
    Iterates through local images, checks if they are in DB, 
    and inserts embeddings if missing.
    """
    print("Starting vector indexing process...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get all processed card_ids to skip
    cur.execute("SELECT card_id FROM card_embeddings")
    existing_ids = set(row[0] for row in cur.fetchall())
    
    # Find all jpg files
    image_files = glob.glob(os.path.join(IMAGE_STORAGE_PATH, "*.jpg"))
    print(f"Found {len(image_files)} images in storage.")
    
    processed_count = 0
    for img_path in image_files:
        try:
            # Filename is "{id}.jpg", so we extract the ID
            basename = os.path.basename(img_path)
            card_id_str = os.path.splitext(basename)[0]
            
            if not card_id_str.isdigit():
                continue
                
            card_id = int(card_id_str)
            
            if card_id in existing_ids:
                continue
            
            # Generate Vector
            vector = vectorizer.vectorize_image(img_path)
            if vector:
                # Save to DB
                cur.execute(
                    "INSERT INTO card_embeddings (card_id, embedding) VALUES (%s, %s)",
                    (card_id, vector)
                )
                conn.commit()
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Indexed {processed_count} new images...")
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            conn.rollback()
            
    cur.close()
    conn.close()
    print(f"Indexing completed. Added {processed_count} new vectors.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
