from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.grpc import Cosine
import torch
import uuid
from rich import print as rprint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
collection_name = "patent_semantic_search"
qdrant_client = QdrantClient(host="localhost", port=6333)

def load_documents():
    # Using chonkie to semantic chunk the markdown text
    from chonkie import SemanticChunker
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        threshold=0.8,
        chunk_size=2048,
        similarity_window=3,
        min_sentences_per_chunk=2,
        min_characters_per_sentence=24
    )

    chunks = chunker(db_load())

    from sentence_transformers import SentenceTransformer

    MODEL = 'multi-qa-mpnet-base-cos-v1'
    embedder = SentenceTransformer(MODEL)
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    #print(embeddings.shape) 

    upsert_chunks_to_vecdb(embeddings, chunks)
    return embeddings,embedder,chunks

def db_load():
    sources = [
    "https://arxiv.org/pdf/2408.09869",
    "https://arxiv.org/pdf/2408.09870",
    "https://arxiv.org/pdf/2408.09871",
    "https://arxiv.org/pdf/2408.09872",
    "https://arxiv.org/pdf/2408.09873",
    "https://arxiv.org/pdf/2408.09874",
    ]

    from docling.document_converter import DocumentConverter
    
    # Loop through all documents and combine their markdown text
    all_markdown_text = ""
    converter = DocumentConverter()
    
    for i, source in enumerate(sources):
        print(f"Processing document {i+1}/{len(sources)}: {source}")
        try:
            result = converter.convert(source)
            markdown_text = result.document.export_to_markdown()
            
            # Add document separator and source info
            all_markdown_text += f"\n\n--- Document {i+1}: {source} ---\n\n"
            all_markdown_text += markdown_text
            
            print(f"✅ Successfully processed document {i+1}")
            
        except Exception as e:
            print(f"❌ Error processing document {i+1} ({source}): {e}")
            continue
    
    print(f"📄 Total documents processed: {len(sources)}")
    print(f"📝 Combined markdown text length: {len(all_markdown_text)} characters")
    
    return all_markdown_text


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

    #Upsert points into Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    print("Chunks embedding shape:", embeddings.shape[-1])
    print("Embeddings stored in Qdrant collection:", collection_name)
    print("Total vetors stores in Qdrant", len(points))

    return (len(points))



if __name__ == "__main__":
    results=load_documents()
    #print(results)