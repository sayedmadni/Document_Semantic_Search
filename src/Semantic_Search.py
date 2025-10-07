#Trying to use GPU
import torch
if torch.cuda.is_available():
    # Use the first available CUDA device
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    # Use CPU if CUDA is not available
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")
# -----------------------------------------------------------Getting the model ready---------------------------------------------------------
# using docling to convert the document to markdown format
from docling.document_converter import DocumentConverter
# At this step we need to go to Anurag directory and take all the documents and loop through them and convert them to markdown format using docling
source = "https://arxiv.org/pdf/2408.09869"  # document path, it can also be a array of document names
converter = DocumentConverter()
result = converter.convert(source) 
markdown_text = result.document.export_to_markdown()
#print(markdown_text)

# Using chonkie to chunk the markdown text
from chonkie import RecursiveChunker
chunker = RecursiveChunker()
chunks = chunker(markdown_text)

for chunk in chunks:
    print(f"Chunk: {chunk.text}\n")

#-----------------------------Next we ge tthe sentence tranformer model ready---------------------------------------------------------
from sentence_transformers import SentenceTransformer

MODEL = 'msmarco-distilbert-base-v4'
embedder = SentenceTransformer(MODEL)
embeddings = embedder.encode(chunks, convert_to_tensor=True) 
# Here the sentences is a list of array of all the documents in the corpus. For us it will be title of each document. 
# This is where we need to use docking and chonkie

# Next we see the shape of the embeddings
print(embeddings.shape)
#print (f'{chunks[0]}  {embeddings[0]}')

# Now we will have input box where the user will enter something and we will search for the most similar document in the corpus. 

query_text = "chunking is fun" # our user input will be this variable
# Extra step to add query cleaning and preprocessing here
query = embedder.encode(query_text, convert_to_tensor=True) # we encode our user input 

# Now we go find the top 3 results that match the most with our query
from sentence_transformers import util
search_results = util.semantic_search(query, embeddings, top_k = 3)
#print(search_results)

# Then we show the documents to the user. 
for index, result in enumerate(search_results[0]):
    print('-'*80)
    print(f'Search Rank: {index}, Relevance score: {result["score"]} ')
    print(chunks[result['corpus_id']])

# Pasting the code here for query cleaning to be used later when stremlit is integrated. 
 # Add query guidelines
    # st = 'streamlit'
    # with st.expander("📝 Query Guidelines", expanded=False):
    #     st.markdown("""
    #     **✅ Good queries:**
    #     - "person with blonde hair and blue eyes"
    #     - "smiling woman wearing glasses"
    #     - "man with beard and mustache"
    #     - "person in formal suit"
    #     - "woman with red hair smiling"
    #     - "person wearing a hat"
        
    #     **❌ Please avoid:**
    #     - Special characters: @#$%^&*+=
    #     - URLs or links
    #     - Script tags or code
    #     - Very long descriptions (>500 chars)
    #     - Obvious inappropriate content
    #     """)

    #     # Input validation and safety checks
    #     def validate_query(query):
    #         """Validate and sanitize user input for safety"""
    #         import re
            
    #         # Remove leading/trailing whitespace
    #         query = query.strip()
            
    #         # Check if query is empty after stripping
    #         if not query:
    #             return None, "Please enter a search query."
            
    #         # Check minimum length
    #         if len(query) < 2:
    #             return None, "Query must be at least 2 characters long."
            
    #         # Check maximum length (prevent extremely long queries)
    #         if len(query) > 500:
    #             return None, "Query is too long. Please keep it under 500 characters."
            
    #         # Basic content filtering - only block obvious inappropriate content
    #         # This is minimal to avoid false positives and over-censorship
    #         inappropriate_patterns = [
    #             # Only block obvious explicit content patterns
    #             r'\b(porn|pornographic|xxx)\b',
    #             # Block obvious hate speech
    #             r'\b(nazi|terrorist)\b'
    #         ]
            
    #         # Check for inappropriate content (minimal filtering)
    #         for pattern in inappropriate_patterns:
    #             if re.search(pattern, query, re.IGNORECASE):
    #                 return None, "Query contains inappropriate content. Please use descriptive text about appearance and clothing."
            
    #         # Remove potentially harmful characters but keep normal text
    #         # Allow letters, numbers, spaces, basic punctuation, and common characters
    #         safe_query = re.sub(r'[^\w\s\-.,!?()\'":;]', '', query)
            
    #         # Check if anything meaningful remains after sanitization
    #         if len(safe_query.strip()) < 2:
    #             return None, "Query contains invalid characters. Please use only letters, numbers, and basic punctuation."
            
    #         # Check for suspicious patterns (basic security)
    #         suspicious_patterns = [
    #             r'<script', r'javascript:', r'on\w+\s*=', r'data:', r'vbscript:',
    #             r'file:', r'ftp:', r'http:', r'https:', r'//', r'\\\\'
    #         ]
            
    #         for pattern in suspicious_patterns:
    #             if re.search(pattern, query, re.IGNORECASE):
    #                 return None, "Query contains potentially unsafe content. Please use only descriptive text."
            
    #         # Additional check for repeated characters (spam-like)
    #         if re.search(r'(.)\1{4,}', query):  # 5 or more repeated characters
    #             return None, "Query contains too many repeated characters. Please use normal text."
            
    #         return safe_query.strip(), None
        
    #     # Validate the query
    #     validated_query, error_msg = validate_query(q)
        
    #     if error_msg:
    #         st.error(f"❌ {error_msg}")
    #         st.stop()
        
    #     # Update the query variable with the validated version
    #     q = validated_query
    #     # Show the processed query for transparency
    #     if q != q.strip():  # Only show if query was modified
    #         st.info(f"🔍 Processed query: '{q}'")