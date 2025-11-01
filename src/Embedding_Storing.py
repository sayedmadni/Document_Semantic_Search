


def embed_chunks(chunks):
    from sentence_transformers import SentenceTransformer

    MODEL = 'multi-qa-mpnet-base-cos-v1'
    embedder = SentenceTransformer(MODEL)
    embeddings = embedder.encode(chunks, convert_to_tensor=True, device=DEVICE)
    #print(embeddings.shape) 

    upsert_chunks_to_vecdb(embeddings, chunks)
    return embeddings,embedder,chunks

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
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vecs[i],
                payload={
                    "type":"chunk",
                    "text":chunks[i],
                    "chunk_idx":i
                }
            )
        )
    
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
    batch_size = 50  # Process 50 points at a time
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
        results=embed_chunks()
        print(results)
