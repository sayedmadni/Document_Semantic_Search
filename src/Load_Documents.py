import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.grpc import Cosine
import torch
import uuid
from rich import print as rprint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
collection_name = "patent_semantic_search"
qdrant_client = QdrantClient(
    host="localhost", 
    port=6333,
    timeout=60  # Increase timeout to 60 seconds
)

if __name__ == "__main__":
    results=load_documents()