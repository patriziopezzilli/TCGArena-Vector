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

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/vectorize/sync")
async def trigger_sync(background_tasks: BackgroundTasks):
    """
    Triggers a background task to scan the image folder and generate vectors 
    for images that haven't been indexed yet.
    """
    background_tasks.add_task(process_batch_images)
    return {"message": "Sync started in background"}

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
    
    # Perform Nearest Neighbor Search using L2 distance (Euclidean)
    # The <-> operator is provided by pgvector for L2 distance
    # Increased from LIMIT 5 to LIMIT 20 for better recall
    cur.execute("""
        SELECT card_id, (embedding <-> %s::vector) as distance
        FROM card_embeddings
        ORDER BY distance ASC
        LIMIT 20;
    """, (query_vector,))

    results = cur.fetchall()
    cur.close()
    conn.close()

    # Convert L2 distance to confidence percentage
    # L2 distance typically ranges from 0 (perfect match) to ~15+ (very different)
    # We map this to 0-100% confidence scale
    def distance_to_confidence(distance: float) -> float:
        """
        Converts L2 distance to confidence percentage (0-100)

        Distance 0    → 100% confidence (perfect match)
        Distance 5    → ~67% confidence (good match)
        Distance 10   → ~33% confidence (poor match)
        Distance 15+  → ~0% confidence (no match)
        """
        # Clamp distance to reasonable range
        clamped = max(0.0, min(distance, 15.0))

        # Invert and normalize to 0-100 scale
        confidence = ((15.0 - clamped) / 15.0) * 100.0

        return round(confidence, 2)

    return [
        {
            "card_id": row[0],
            "confidence": distance_to_confidence(float(row[1]))
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
