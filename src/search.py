from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
collection_name = "patent_semantic_search"
qdrant_client = QdrantClient(host="localhost", port=6333)
MODEL = 'multi-qa-mpnet-base-cos-v1'
embedder = SentenceTransformer(MODEL)


def search_qdrant(q:str) -> list[str]:
    
#     text_query = "What is the relationship between force and acceleration?"
# matryoshka_768_vectors = matryoshka_model.encode([text_query], convert_to_numpy=True)[0]
# response = qdrant_client.query_points(
#             collection_name="hybrid_search",
#             query=matryoshka_768_vectors.tolist(),
#             using="matryoshka_768",
#             with_payload=True,
#             limit=3,
#         )

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
    for p in search_result:
        if p.payload and "text" in p.payload:
            texts.append(p.payload["text"])
    return texts

if __name__ == "__main__":
    results=search_qdrant("What is chonkie?")
    print(results)
    