from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DocItemLabel
from datetime import datetime

#from docling.pipeline.standard_pipeline import StandardPipeline

# The connection URI, with the 'admin' user and password you set in the docker run command.
# 'localhost' refers to your machine, which Docker maps to the container's port.
MONGO_URI = "mongodb://admin:llm_raptor_123@localhost:27017/"

def load_chunks_in_db(text_chunks, non_text_chunks):   
    try:
        """
        Create a folder to store chunks and save them as JSON files
        """
        
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["doc_college_pathway"]
        collection = db["college_pathway"]

        print("inserting in mongodb")
        collection.insert_many(text_chunks)
        collection.insert_many(non_text_chunks)
       
        print("========================Insertion suucesful ==========")

        print(f"Successfully inserted {len(text_chunks)} chunks into MongoDB.")
    except Exception as e:
        print(f"Exception occured while MongoDB operation: {e}")
    finally:
        # Close the connection when done
        if 'client' in locals() and client:
            client.close()  

#all the below code are for creating chunks from single pdf and loading in mongodb
def read_from_mongodb_test():
    try:
        # Create a new client instance
        client = MongoClient(MONGO_URI)

        # Test the connection to the server
        client.admin.command('ping')
        print("Connection to MongoDB successful!")

        # Access the database and collection
        db = client.testdb
        collection = db.users

        # --- Read all documents in the 'users' collection ---
        print("\nReading all users:")
        for user in collection.find():
            print(user)

        # --- Read a specific document ---
        print("\nReading one user named 'Jane Doe':")
        jane_doe = collection.find_one({"name": "Jane Doe"})
        print(jane_doe)

    except ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    finally:
        # Close the connection when done
        if 'client' in locals() and client:
            client.close()


def load_pdf_into_mongo():

    # The source can be a local file path or a URL
    source = "https://arxiv.org/pdf/2408.09869"

    # Initialize the converter and process the document
    converter = DocumentConverter()
    result = converter.convert(source)

    # Print the clean, structured Markdown output
    print(result.document.export_to_markdown())

def load_pdf_mongodb_():
    try:
        from pathlib import Path
        
        # ---------- Step 1: Read files from education directory ----------
        education_dir = "/home/anuragd/labshare/corpus/education"
        education_path = Path(education_dir)
        
        # Get all PDF files in the directory (pick only first 25)
        all_pdfs = list(education_path.glob("*.pdf"))
        pdf_files = all_pdfs[:10]
        print(f"Found {len(all_pdfs)} PDF files in {education_dir}, processing first 25")
        
        # Connect to MongoDB (do this once before the loop)
        client = MongoClient(MONGO_URI)
        db = client["doc_college_db"]
        collection = db["text_chunks"]
        
        # Loop through each PDF file
        for pdf_file in pdf_files:
            try:
                source = str(pdf_file)
                print(f"\nProcessing file: {pdf_file.name}")
                
                converter = DocumentConverter()
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
                        #print(f"Non-text chunks: {chunk.text}")
                    else:
                        text_chunks.append(chunk)
                        #print(f"Text chunk: {chunk.text}")
                
                # Create chunks folder and save chunks
                #create_chunks_folder_and_save(text_chunks, non_text_chunks, filename)
                    

                # ---------- Step 3: Store chunks in MongoDB ----------
                print(f"Inserting {len(text_chunks)} chunks from {pdf_file.name} into MongoDB")

                # Insert chunks
                for i, chunk in enumerate(text_chunks):
                    collection.insert_one({
                        "chunk_id": i + 1,
                        "text": chunk.text,
                        "source_url": source,
                        "filename": pdf_file.name
                    })
                print(f"========================Insertion successful for {pdf_file.name} ==========")
                print(f"Successfully inserted {len(text_chunks)} chunks from {pdf_file.name} into MongoDB.")
            except Exception as file_error:
                print(f"❌ Error processing file {pdf_file.name}: {file_error}")
                print(f"⚠️ Skipping file {pdf_file.name} and continuing with next file...")
                continue
        
        print(f"\n=== Completed processing all {len(pdf_files)} files ===")
        
    except Exception as e:
        print(f"Exception occured while MongoDB operation: {e}")
    finally:
        # Close the connection when done
        if 'client' in locals() and client:
            client.close()

def chunk_pdf():
    try:
        from pathlib import Path
        
        # ---------- Step 1: Read files from education directory ----------
        education_dir = "/home/anuragd/labshare/corpus/education"
        education_path = Path(education_dir)
        
        # Check if directory exists
        if not education_path.exists():
            raise FileNotFoundError(f"Directory not found: {education_dir}")
        
        # Get all PDF files in the directory (pick only first 25)
        all_pdfs = list(education_path.glob("*.pdf"))
        pdf_files = all_pdfs[:25]
        print(f"Found {len(all_pdfs)} PDF files in {education_dir}, processing first 25")
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {education_dir}")
        
        # Initialize converter and chunker
        converter = DocumentConverter()
        chunker = HybridChunker()
        
        # Accumulate all chunks from all files
        all_text_chunks_data = []
        all_non_text_chunks_data = []
        
        # Loop through each PDF file
        for pdf_file in pdf_files:
            try:
                source = str(pdf_file)
                print(f"\nProcessing file: {pdf_file.name}")

                result = converter.convert(source).document
                # markdown_text = result.document.export_to_markdown()
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
                
                # Save text chunks
                if text_chunks:
                    for i, chunk in enumerate(text_chunks):
                        chunk_data = {
                            "chunk_id": f"text_{pdf_file.stem}_{i+1}",
                            "text": chunk.text,
                            "metadata": {
                                "chunk_type": "text",
                                "created_at": datetime.now().isoformat(),
                                "source_document": source,
                                "filename": pdf_file.name
                            }
                        }
                        all_text_chunks_data.append(chunk_data)

                # Save non-text chunks
                if non_text_chunks:
                    for i, chunk in enumerate(non_text_chunks):
                        chunk_data = {
                            "chunk_id": f"non_text_{pdf_file.stem}_{i+1}",
                            "text": chunk.text,
                            "metadata": {
                                "chunk_type": "non_text",
                                "created_at": datetime.now().isoformat(),
                                "source_document": source,
                                "filename": pdf_file.name
                            }
                        }
                        all_non_text_chunks_data.append(chunk_data)
                
                print(f"Processed {len(text_chunks)} text chunks and {len(non_text_chunks)} non-text chunks from {pdf_file.name}")
            except Exception as file_error:
                print(f"❌ Error processing file {pdf_file.name}: {file_error}")
                print(f"⚠️ Skipping file {pdf_file.name} and continuing with next file...")
                continue
        
        # Load all chunks in mongodb
        if all_text_chunks_data or all_non_text_chunks_data:
            print(f"\nPreparing to insert into MongoDB...")
            print(f"Total text chunks to insert: {len(all_text_chunks_data)}")
            print(f"Total non-text chunks to insert: {len(all_non_text_chunks_data)}")
            load_chunks_in_db(all_text_chunks_data, all_non_text_chunks_data)
            print(f"\n=== Completed processing all {len(pdf_files)} files ===")
            print(f"Total text chunks: {len(all_text_chunks_data)}")
            print(f"Total non-text chunks: {len(all_non_text_chunks_data)}")
        else:
            print("No chunks to insert - both text and non-text chunks lists are empty")
            

    except Exception as e:
        print(f"Exception occured while MongoDB operation: {e}")


if __name__ == "__main__":
    results=chunk_pdf()
    #load_pdf_into_mongo()
