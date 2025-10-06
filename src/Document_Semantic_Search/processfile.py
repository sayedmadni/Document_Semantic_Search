import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from chonkie import RecursiveChunker, SentenceTransformerEmbeddings
from ollama import Client
import numpy as np

# Step 0: Initial setup
PDF_PATH = "dummy_for_docling.pdf"


# Step 1 & 2: Parse, chunk, and embed the PDF
def process_document(pdf_path: str):
    """Parses, chunks, and embeds a PDF file, returning an in-memory list."""
    # Parse PDF using Docling
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    markdown_content = result.document.export_to_markdown()
    print("markdown_content conversion done-----------")
    # Chunk with Chonkie
    chunker = RecursiveChunker()
    chunks = chunker(markdown_content)
    print("markdown_content chunks  done-----------")

    # Embed with Sentence-Transformers using Chonkie's helper
    print("Generating embeddings...")
    ef = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = ef.embed_batch(chunk_texts)

    # Store data in a list of dictionaries (in-memory)
    in_memory_store = [
        {"text": chunk.text, "vector": embedding, "id": i}
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    print(f"Stored {len(in_memory_store)} chunks in memory.")
    return in_memory_store, ef

# Step 4: Retrieve relevant chunks from memory
def retrieve_context(query, in_memory_store, ef, k=3):
    """Retrieves top-k relevant chunks based on a query using cosine similarity."""
    query_embedding = ef.embed(query)
    
    # Calculate cosine similarity for all embeddings
    similarities = []
    for item in in_memory_store:
        item_vector = np.array(item["vector"])
        # Use dot product for cosine similarity with normalized vectors
        similarity = np.dot(query_embedding, item_vector)
        similarities.append((similarity, item["text"]))

    # Sort and retrieve top-k
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k_chunks = [item[1] for item in similarities[:k]]
    
    context = "\n\n".join(top_k_chunks)
    print(f"\nRetrieved {len(top_k_chunks)} chunks for the query.")
    return context

# Step 5: Answer the question using the LLM
def answer_question_with_llm(query, context):
    """Uses Ollama to generate an answer based on the query and context."""
    client = Client()
    prompt = f"""
    You are an AI assistant tasked with answering questions based on the provided context.
    Use the context below to answer the user's query. If the answer cannot be found in the context,
    politely state that the information is not available in the document.

    Context:
    {context}

    Query:
    {query}

    Answer:
    """
    response = client.generate(model='llama3.1', prompt=prompt)
    return response['response']

# Main RAG pipeline with in-memory storage
if __name__ == "__main__":
    try:
        # Step 1-3: Process document and store in memory
        pdf_url = "https://arxiv.org/pdf/2304.03442"

        in_memory_store, embedder = process_document(pdf_url)

        # User interaction loop
        while True:
            user_query = input("\nEnter your question (or 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break

            # Step 4: Retrieve context from in-memory store
            context = retrieve_context(user_query, in_memory_store, embedder)

            # Step 5: Answer question with LLM
            answer = answer_question_with_llm(user_query, context)
            print("-" * 50)
            print(f"**Answer:** {answer}")
            print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if Path(PDF_PATH).is_file():
            os.remove(PDF_PATH)
