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
        # ---------- Step 1: Read the file from an online URL ----------
        source = "https://levelupcoalition.org/wp-content/uploads/2017/11/The-Case-for-Mathematics-Pathways-from-the-Launch-Years-in-High-School-through-Postsecondary-Education.pdf"  # Example file
        

        print("File downloaded successfully.")
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
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["doc_college_db"]
        collection = db["text_chunks"]

        print("inserting in mongodb")

        # Insert chunks
        for i, chunk in enumerate(text_chunks):
            collection.insert_one({
                "chunk_id": i + 1,
                "text": chunk.text,
                "source_url": source
            })
        print("========================Insertion suucesful ==========")

        print(f"Successfully inserted {len(text_chunks)} chunks into MongoDB.")
    except Exception as e:
        print(f"Exception occured while MongoDB operation: {e}")
    finally:
        # Close the connection when done
        if 'client' in locals() and client:
            client.close()



def chunk_pdf():
    try:
        # ---------- Step 1: Read the file from an online URL ----------
        source = "https://levelupcoalition.org/wp-content/uploads/2017/11/The-Case-for-Mathematics-Pathways-from-the-Launch-Years-in-High-School-through-Postsecondary-Education.pdf"  # Example file
        

        print("File downloaded successfully.")
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
                print(f"Non-text chunks: {chunk.text}")
            else:
                text_chunks.append(chunk)
                print(f"Text chunk: {chunk.text}")
        
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
                        "source_document": source
                    }
                }
                text_chunks_data.append(chunk_data)

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
                        "source_document": source
                    }
                }
            non_text_chunks_data.append(chunk_data)
        # Load in chunks in mongodb
        load_chunks_in_db(text_chunks_data, non_text_chunks_data)
            

       
    except Exception as e:
        print(f"Exception occured while MongoDB operation: {e}")
   




if __name__ == "__main__":
    results=chunk_pdf()
    #load_pdf_into_mongo()
