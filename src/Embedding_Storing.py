from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import torch
import uuid

# MongoDB configuration (matching recontext_chunking.py)
MONGO_URI = "mongodb://admin:llm_raptor_123@localhost:27017/"
DB_NAME = "doc_college_db"
RECONTEXTUALIZED_COLLECTION = "recontextualized_chunks"

# Qdrant configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
collection_name = "college_pathway_search"
qdrant_client = QdrantClient(
    host="localhost", 
    port=6333,
    timeout=60  # Increase timeout to 60 seconds
)

def load_recontextualized_chunks_from_db():
    """
    Load recontextualized chunks from MongoDB collection.
    Based on load_recontextualized_chunks_in_db in recontext_chunking.py.
    
    Returns
    -------
    List[str]: A list of contextual_chunk text strings ready for embedding.
    """
    print("[INFO] Loading recontextualized chunks from MongoDB...")
    
    client = None
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[RECONTEXTUALIZED_COLLECTION]
        
        # Fetch all documents from the collection
        cursor = collection.find()
        recontextualized_chunks = []
        
        for doc in cursor:
            # Extract the contextual_chunk text field
            contextual_text = doc.get("contextual_chunk", "").strip()
            if contextual_text:
                recontextualized_chunks.append(contextual_text)
        
        print(f"[OK] Successfully loaded {len(recontextualized_chunks)} recontextualized chunks from MongoDB.")
        return recontextualized_chunks
        
    except Exception as e:
        print(f"[ERROR] Exception occurred while loading from MongoDB: {e}")
        raise
    finally:
        # Close the connection when done
        if client:
            client.close()

def embed_chunks():
    """
    Embed chunks into vectors and store in Qdrant.
    
    Returns
    -------
    tuple: (embeddings, embedder, chunks)
    """
    from sentence_transformers import SentenceTransformer

    # Load from MongoDB if chunks not provided
    chunks = load_recontextualized_chunks_from_db()
    
    if not chunks or len(chunks) == 0:
        raise ValueError("No chunks available to embed. Either provide chunks or ensure MongoDB collection has data.")
    
    MODEL = 'multi-qa-mpnet-base-cos-v1'
    embedder = SentenceTransformer(MODEL)
    embeddings = embedder.encode(chunks, convert_to_tensor=True, device=DEVICE)
    #print(embeddings.shape) 

    upsert_chunks_to_vecdb(embeddings, chunks)
    return embeddings, embedder, chunks

def upsert_chunks_to_vecdb(embeddings, chunks):
    # check docker ps 
    # uncomment and run from labshare if the docker instance is not running
    # docker run -d -p 0.0.0.0:6333:6333 -p 6334:6334 \
    # -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    # qdrant/qdrant

    # print("Connection with qdrant port 6333 established")
    
    #connection name

    #derive vector size from embeddings
    dim=embeddings.shape[-1]
    print("embeddings shape, hence the desired size of the vector collection, is ", dim)

    # We should never create the embeddings on run time. It should be pre-loaded and stored in the database.
    try:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name=collection_name)
            print(f"Older collection {collection_name} deleted")
    except Exception as e:
        print(f"Warning: Could not check/delete existing collection: {e}")

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE))
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise
    
    print(f"New collection: {collection_name} created")

    #Prepare points for vector upsert
    points = []
    
    # If embeddings is a torch.Tensor: move to CPU and to list-of-lists
    try:
        vecs = embeddings.cpu().tolist()
    except AttributeError:
        # if it's already a list/np array on CPU
        vecs = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    for i in range(len(vecs)):
        # print(f"for loop embedding # {i}")
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vecs[i],
            payload={
                "type":"chunk",
                "text":chunks[i],
                "chunk_idx":i
            }
        )
        points.append(point)
        
        #Print point contents
        #print(f"\nPoint {i} contents:")
        #print(f"  ID: {point.id}")
        #print(f"  Vector shape: {len(point.vector)} dimensions")
        #print(f"  Vector (first 5 values): {point.vector[:5]}")
        #print(f"  Payload: {point.payload}")
        #print(f"  Chunk text (first 100 chars): {chunks[i][:100]}...")
    
    # this can be used to clean up the chunks if needed. 
    # def ensure_texts(chunks):
    # # chonkie chunks often have `.text`; if already strings, just return them
    # if len(chunks) == 0:
    #     return []
    # if isinstance(chunks[0], str):
    #     return chunks
    # # fallback: objects with `.text`
    # return [c.text for c in chunks]

    #Upsert points into Qdrant in batches to avoid timeout
    batch_size = 50  
    # Process 50 points at a time
    total_points = len(points)
    
    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        print(f"Upserting batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size} ({len(batch)} points)")
        
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"✅ Successfully upserted batch {i//batch_size + 1}")
        except Exception as e:
            print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
            # Continue with next batch instead of failing completely
            continue

    print("Chunks embedding shape:", embeddings.shape[-1])
    print("Embeddings stored in Qdrant collection:", collection_name)
    print("Total vetors stores in Qdrant", len(points))

    return (len(points))

if __name__ == "__main__":
    # Call embed_chunks() without arguments to load from recontextualized_chunks collection
    results = embed_chunks()
    print(results)
