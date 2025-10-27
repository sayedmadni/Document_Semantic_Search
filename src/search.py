from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
collection_name = "patent_semantic_search"
qdrant_client = QdrantClient(host="localhost", port=6333)
MODEL = 'multi-qa-mpnet-base-cos-v1'
embedder = SentenceTransformer(MODEL)


def search_qdrant(q:str) -> list[str]:
    # Encode the query into a vector list[float]
    query = embedder.encode([q], convert_to_numpy=True)[0].tolist()

    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query,
        with_payload=True,
        limit=3
    ).points

    # Return just the payload texts (if present)
    texts = []
    for index,p in enumerate(search_result):
        if p.payload and "text" in p.payload:
            texts.append(f"Result {index+1}: {p.payload["text"]["text"]}")
    print("\n".join(texts))
    return texts

def search_qdrant_queries(queries: list[str]) -> list[str]:
    # Encode the query into a vector list[float]
    query_vectors = embedder.encode(queries, convert_to_numpy=True)

    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vectors,
        with_payload=True,
        limit=3
    ).points

    # Return just the payload texts (if present)
    texts = []
    for index,p in enumerate(search_result):
        if p.payload and "text" in p.payload:
            texts.append(f"Result {index+1}: {p.payload["text"]["text"]}")
    print("\n".join(texts))
    return texts


if __name__ == "__main__":
    results=search_qdrant("What is chonkie?")
    #print(results)
    