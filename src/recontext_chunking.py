import os
import json
from typing import List, Dict, Any
from pathlib import Path
from pymongo import MongoClient
from chonkie import SemanticChunker
from openai import OpenAI

# ============================================================
# recontext_chunking.py
# ============================================================
# This script performs the *middle layer* of a RAG pipeline:
#   1. Extract text chunks from MongoDB
#   2. Run semantic chunking using Chonkie
#   3. Run contextual chunking using an Ollama-served LLM
#   4. Produce JSONL outputs for downstream embedding
#
# Each function is written to be simple, transparent, and testable.
# ============================================================


# -------------------------
# Configuration
# -------------------------

# --- Mongo ---
MONGO_URI = "mongodb://admin:llm_raptor_123@localhost:27017/"

# docker exec -it  llm-raptor-mongodb mongosh -u admin -p llm_raptor_123 --authenticationDatabase admin
# use doc_collelsge_db use this command to switch to the database
# db.[collection_name].find() Use this commant to view all data in the collection 

DB_NAME = "doc_college_db"
SRC_COLLECTION = "text_chunks"  # upstream loaded here

# --- File outputs ---
BOOTCAMP_ROOT_DIR = Path("/home/anuragd/labshare/")  # hardcoded absolute path
OUT_DIR = BOOTCAMP_ROOT_DIR / "doc_sem_srch/chunking/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # create folder if not there

SEMANTIC_JSONL = OUT_DIR / "semantic_chunks.jsonl"
CONTEXTUAL_JSONL = OUT_DIR / "contextual_chunks.jsonl"

# --- Ollama (OpenAI-compatible endpoint) ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # dummy key required by OpenAI client
CTX_MODEL = "gpt-oss:20b"  # model served by Ollama


# ============================================================
# 1) Extract parent chunks from MongoDB
# ============================================================

def extract_parent_chunks_from_mongo() -> List[Dict[str, Any]]:
    """
    Extract parent text chunks from MongoDB.

    Logic
    -----
    - Connects to the MongoDB collection.
    - Filters only documents where 'type' == 'text'.
    - Extracts the 'content' field as text.
    - Returns a list of dicts: [{"id": int, "text": str}, ...]

    Assumptions
    -----------
    - Your upstream process inserts documents like:
        {"chunk_id": 1, "type": "text", "content": "AI education pathway intro"}
    - So this function ignores metadata and focuses on text chunks only.
    """
    print("[TEST] Starting: extract_parent_chunks_from_mongo()")

    client = MongoClient(MONGO_URI)
    parent_chunks = []
    try:
        coll = client[DB_NAME][SRC_COLLECTION]
        cursor = coll.find()
        # print(f"cursor: {cursor}")
        for i, doc in enumerate(cursor):
            # print(f"doc: {doc}")
            txt = (doc.get("text") or "").strip()
            if not txt:
                continue
            parent_chunks.append({"id": i, "text": txt})
        print(f"[OK] Total parent chunks extracted: {len(parent_chunks)}")
    finally:
        client.close()

    return parent_chunks


# ============================================================
# 2) Semantic chunking (Chonkie)
# ============================================================

def create_semantic_chunks(parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produce *semantic* chunks from parent texts using Chonkie.

    Inputs
    ------
    parent_chunks : [{"id": int, "text": str}, ...]
        - Each dict represents one parent text block.
        - Assumes text is already filtered and non-empty.

    Output
    ------
    Returns a flat list for the next step:
      [{
        "chunk_id": int,                 # index within its parent
        "chunk": <Chonkie span object>,  # has .text, .start_index, .end_index
        "parent_id": int,
        "parent_chunk": {"id": int, "text": str}
      }, ...]

    Side Effect
    -----------
    Writes a JSONL file (semantic_chunks.jsonl) for inspection / reproducibility.
    This provides a permanent record of how texts were segmented.
    """
    print("[TEST] Starting: create_semantic_chunks()")
    print(f"[TEST] Writing semantic chunks to: {SEMANTIC_JSONL}")

    # Use Chonkie's default semantic model.
    # The goal is to capture meaning-coherent spans, not fixed sizes.
    semantic_chunker = SemanticChunker()
    semantic_chunks: List[Dict[str, Any]] = []
    total_semantic = 0

    with SEMANTIC_JSONL.open("w", encoding="utf-8") as f:
        for p in parent_chunks:
            text = (p.get("text") or "").strip()
            if not text:
                continue

            # Run Chonkie to get semantic spans
            sem_spans = semantic_chunker.chunk(text)

            # Assign ordered IDs and store results
            for chunk_id, sc in enumerate(sem_spans):

                # --- JSONL artifact ---
                # Lightweight line-by-line record for traceability and reuse.
                f.write(json.dumps({
                    "parent_id": p["id"],
                    "chunk_id": chunk_id,
                    "text": sc.text,
                    "start_char": sc.start_index,
                    "end_char": sc.end_index
                }, ensure_ascii=False) + "\n")

                # --- In-memory representation ---
                # Keeps the full Chonkie object for downstream contextualization.
                semantic_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk": sc,
                    "parent_id": p["id"],
                    "parent_chunk": p
                })
                total_semantic += 1

    print(f"[OK] Total semantic chunks created: {total_semantic}")
    return semantic_chunks


# ============================================================
# 3) Contextual chunking (Ollama LLM)
# ============================================================

def create_contextual_chunks(semantic_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add minimal parent context to each semantic chunk so that it stands alone.

    Logic
    -----
    - For each semantic chunk, send a prompt to an Ollama-served model
      using the OpenAI API format.
    - The model enriches the chunk by adding only as much context from
      its parent text as needed for self-containment.
    - Returns contextualized text ready for embedding.

    Output
    ------
    A list of dicts:
      [{
        "contextual_chunk": str,
        "semantic_chunk_id": int,
        "parent_id": int,
        "semantic_chunk": str,
        "parent_chunk": str
      }, ...]

    Side Effect
    -----------
    Writes a contextual_chunks.jsonl file — one enriched chunk per line.
    """
    print("[TEST] Starting: create_contextual_chunks()")
    print(f"[TEST] Using Ollama model: {CTX_MODEL}")
    print(f"[TEST] Writing contextual chunks to: {CONTEXTUAL_JSONL}")

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    contextual_chunks: List[Dict[str, Any]] = []
    total_contextual = 0

    # The guiding system prompt for the LLM.
    system_prompt = (
        "Here is the parent section: {parent_text}. "
        "Now here is the semantic chunk: {sc_text}. "
        "Please produce an enriched chunk which retains the semantic chunk "
        "but adds any necessary context from the parent so that the chunk is self-standing.\n"
        "Do not add anything that is not needed to make the chunk self-standing."
    )

    with CONTEXTUAL_JSONL.open("w", encoding="utf-8") as f:
        for sc in semantic_chunks:
            parent_text = sc["parent_chunk"]["text"]
            sc_text = sc["chunk"].text

            try:
                # --- Core model call ---
                # Sends a structured prompt to Ollama (via OpenAI client).
                resp = client.chat.completions.create(
                    model=CTX_MODEL,
                    messages=[{"role": "user",
                               "content": system_prompt.format(
                                   parent_text=parent_text, sc_text=sc_text)}],
                    temperature=0.2,   # deterministic
                    max_tokens=800
                )
                ctx_text = resp.choices[0].message.content or sc_text
            except Exception as e:
                # --- Defensive fallback ---
                print(f"[WARN] Contextualization failed for parent={sc['parent_id']} "
                      f"chunk={sc['chunk_id']}: {e}")
                ctx_text = sc_text

            # Combine outputs into a single record.
            row = {
                "contextual_chunk": ctx_text,
                "semantic_chunk_id": sc["chunk_id"],
                "parent_id": sc["parent_id"],
                "semantic_chunk": sc_text,
                "parent_chunk": parent_text
            }

            # Write to disk and keep in memory for embedding.
            contextual_chunks.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_contextual += 1

    print(f"[OK] Total contextual chunks created: {total_contextual}")
    print(type(contextual_chunks))
    return contextual_chunks


def load_recontextualized_chunks_in_db(contextual_chunks):
   
    try:
        
        DB_NAME = "doc_college_db"
        SRC_COLLECTION = "recontextualized_chunks"  # upstream loaded here
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[SRC_COLLECTION]

        print("inserting recontextualized chunks in mongodb")
        collection.insert_many(contextual_chunks)
       
        print("========================Insertion suucesful ==========")
        print(f"Successfully inserted {len(contextual_chunks)} chunks into MongoDB.")
    except Exception as e:
        print(f"Exception occured while MongoDB operation: {e}")
    finally:
        # Close the connection when done
        if 'client' in locals() and client:
            client.close()  


# ============================================================
# Main entry point
# ============================================================

def main():
    """
    Execute all three steps sequentially:
      1. Extract text chunks from MongoDB
      2. Create semantic chunks via Chonkie
      3. Contextualize each chunk via Ollama
    Prints simple counts at each stage and writes JSONL outputs.
    """
    print("=== Running recontext_chunking.py ===")

    # Step 1: Extract from Mongo
    parents = extract_parent_chunks_from_mongo()

    # Step 2: Semantic chunking
    semantic = create_semantic_chunks(parents)

    # Step 3: Contextual chunking
    contextual = create_contextual_chunks(semantic)

    # Step 4: Load recontextualized chunks in MongoDB
    load_recontextualized_chunks_in_db(contextual)

    # Final confirmation
    print("=== Pipeline complete ===")
    print(f"[OK] Parent chunks extracted: {len(parents)}")
    print(f"[OK] Semantic chunks created: {len(semantic)}")
    print(f"[OK] Contextual chunks created: {len(contextual)}")
    print(f"[OK] Semantic JSONL file: {SEMANTIC_JSONL}")
    print(f"[OK] Contextual JSONL file: {CONTEXTUAL_JSONL}")


if __name__ == "__main__":
    main()
