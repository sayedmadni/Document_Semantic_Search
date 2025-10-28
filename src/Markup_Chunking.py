from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import os
import glob
import json
from datetime import datetime
from docling_core.types.doc import DocItemLabel

def db_load():
    print(f"Hello from db_load")    
    # Local directory containing education documents
    education_dir = "/home/anuragd/labshare/corpus/education"
    # "/home/anuragd/labshare/corpus/markdown/education"
    
    # Find all PDF files in the directory
    pdf_pattern = os.path.join(education_dir, "*.pdf")
    sources = glob.glob(pdf_pattern)
    
    # If no PDFs found, also check for other document types
    if not sources:
        # Check for other common document formats
        for ext in ["*.docx", "*.doc", "*.txt", "*.md"]:
            pattern = os.path.join(education_dir, ext)
            sources.extend(glob.glob(pattern))
    
    print(f"📁 Found {len(sources)} documents in {education_dir}")
    for i, source in enumerate(sources):
        print(f"  {i+1}. {os.path.basename(source)}")
    
    if not sources:
        print(f"❌ No documents found in {education_dir}")
        return ""
    
    # Loop through all documents and combine their markdown text
    all_markdown_text = ""
    converter = DocumentConverter()
    
    for i, source in enumerate(sources[:5]):
        filename = os.path.basename(source)
        print(f"Processing document {i+1}/{len(sources)}: {filename}")
        try:
            result = converter.convert(source).document
            # markdown_text = result.document.export_to_markdown()
            chunker = HybridChunker()
            chunks = list(chunker.chunk(dl_doc=result))
            print(f"Total chunks: {len(chunks)}")
            # for chunk in chunks:
            #     print(chunk.text)

            non_text_chunks = []
            text_chunks = []
            for chunk in chunks:                
                if any(item.label != DocItemLabel.TEXT for item in chunk.meta.doc_items):
                    non_text_chunks.append(chunk)
                    print(f"Non-text chunks: {chunk.text}")
                else:
                    text_chunks.append(chunk)
                    print(f"Text chunk: {chunk.text}")
            
            # Create chunks folder and save chunks
            create_chunks_folder_and_save(text_chunks, non_text_chunks, filename)
        
        except Exception as e:
            print(f"❌ Error processing document {i+1} ({filename}): {e}")
            continue

def create_chunks_folder_and_save(text_chunks, non_text_chunks, filename):
    """
    Create a folder to store chunks and save them as JSON files
    """
    # Create chunks directory in the shared location
    chunks_dir = "/home/anuragd/labshare/corpus/education/chunks"
    
    # Create chunks directory if it doesn't exist
    os.makedirs(chunks_dir, exist_ok=True)
    print(f"📁 Created/verified chunks directory: {chunks_dir}")
    
    # Create a subdirectory for this document
    doc_name = os.path.splitext(filename)[0]  # Remove file extension
    doc_chunks_dir = os.path.join(chunks_dir, doc_name)
    os.makedirs(doc_chunks_dir, exist_ok=True)
    print(f"📁 Created document-specific directory: {doc_chunks_dir}")
    
    # Save text chunks
    if text_chunks:
        text_chunks_data = []
        for i, chunk in enumerate(text_chunks):
            chunk_data = {
                "chunk_id": f"text_{i+1}",
                "text": chunk.text,
                "metadata": {
                    "chunk_type": "text",
                    "created_at": datetime.now().isoformat(),
                    "source_document": filename
                }
            }
            text_chunks_data.append(chunk_data)
        
        text_chunks_file = os.path.join(doc_chunks_dir, "text_chunks.json")
        with open(text_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(text_chunks_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(text_chunks)} text chunks to: {text_chunks_file}")
    
    # Save non-text chunks
    if non_text_chunks:
        non_text_chunks_data = []
        for i, chunk in enumerate(non_text_chunks):
            chunk_data = {
                "chunk_id": f"non_text_{i+1}",
                "text": chunk.text,
                "metadata": {
                    "chunk_type": "non_text",
                    "created_at": datetime.now().isoformat(),
                    "source_document": filename
                }
            }
            non_text_chunks_data.append(chunk_data)
        
        non_text_chunks_file = os.path.join(doc_chunks_dir, "non_text_chunks.json")
        with open(non_text_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(non_text_chunks_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(non_text_chunks)} non-text chunks to: {non_text_chunks_file}")
    
    # Create a summary file
    summary = {
        "document_name": filename,
        "total_text_chunks": len(text_chunks),
        "total_non_text_chunks": len(non_text_chunks),
        "total_chunks": len(text_chunks) + len(non_text_chunks),
        "processed_at": datetime.now().isoformat(),
        "chunks_directory": doc_chunks_dir
    }
    
    summary_file = os.path.join(doc_chunks_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Created summary file: {summary_file}")

if __name__ == "__main__":
    results=db_load()