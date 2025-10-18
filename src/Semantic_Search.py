#Trying to use GPU
from qdrant_client.grpc import Cosine
import torch
import sys
from pathlib import Path
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from check_toxicity import query_guadrails
from check_basic_rules import query_guadrails
from prompt_paraphase import paraphrase_sentence


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# This is where Anurag will create the Payload into qdrantdb item 8 and 9 on google doc
# https://qdrant.tech/documentation/quickstart/
def db_load():
    from docling.document_converter import DocumentConverter
    source = "https://arxiv.org/pdf/2408.09869"  # document path, it can also be a array of document names
    converter = DocumentConverter()
    result = converter.convert(source) 
    markdown_text = result.document.export_to_markdown()
    #print(markdown_text)
    return markdown_text

# This is where Anurag will retrieve the data from qdrantdb with the vectorized input query
def db_retrieve():
    return("Hello")


def upsert_chunks_to_vecdb(embeddings, chunks):
    #make connection
    qdrant_client = QdrantClient(host="localhost", port=6333)
    
    #connection name
    collection_name = "patent_semantic_search"

    #derive vector size from embeddings
    dim=embeddings.shape[-1]

   # Always delete the old collection if it exists and create a new one
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name=collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
        size=dim,
        # TODO: need to confirm with the model whether another method may be need DOT product and/or normalization 
        distance=Distance.Cosine))

    #Prepare points for vector upsert
    points = []
    
    # If embeddings is a torch.Tensor: move to CPU and to list-of-lists
    try:
        vecs = embeddings.cpu().tolist()
    except AttributeError:
        # if it's already a list/np array on CPU
        vecs = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    for i in range(len(vecs)):
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


def load_model_and_data():
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

    #for chunk in chunks:
    #   print(f"Chunk: {chunk.text}\n")

    #-----------------------------Next we get the sentence transformer model ready with GPU acceleration---------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    MODEL = 'multi-qa-mpnet-base-cos-v1'
    embedder = SentenceTransformer(MODEL)
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    #print(embeddings.shape) 
    return embeddings,embedder,chunks

def run_search_app():
    """Launch the Streamlit search interface."""
    embeddings, embedder,chunks = load_model_and_data()
    print(f"✅ Model and data loaded")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%); 
                border: 3px solid #7FB3D3; border-radius: 25px; 
                padding: 1.5rem; margin-bottom: 1rem; 
                box-shadow: 0 20px 40px rgba(127, 179, 211, 0.1);">
        <div style="text-align: center;">
            <h1 style="margin: 0 0 0.5rem 0; background: linear-gradient(90deg, #7FB3D3, #98D8C8, #7FB3D3); 
                       background-size: 200% 200%; -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; background-clip: text; 
                       font-size: 3rem; font-weight: 700; animation: titleShimmer 2s ease-in-out infinite;">
                🔍 Patent Search AI
            </h1>
            <h3 style="margin: 0; color: #2C3E50; font-style: italic; font-size: 1.2rem;">
                *Discover patents through AI-powered search*
            </h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    # Search section with enhanced styling
    st.markdown("## 🔍 AI-Powered Search")
    st.markdown("**Describe what you're looking for and let AI find the perfect patent match!**")
    
    # Initialize session state for search query
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # User input with better layout - spread horizontally
    col1,col2 = st.columns([5,5])
    
    with col1:
        q = st.text_input(
            "🎯 Search Query", 
            value=st.session_state.search_query,
            placeholder="e.g., 'How can I chunk documents'",
            help="Describe the patent info you're looking for in natural language",
            key="search_input"
        ) 

    # Perform search when user enters a query
    if q:
        # Input validation and safety checks
        # This is where Sangeetha will do the Ensamble Prompting / Query Guardrails item 11 on google doc

        if not validate_input(q: str)-> bool:
            return False
            
       

       
        
        with st.spinner("Searching..."):
            try:
                print(f'Query: {q}')
                #getting paraphrase questions for the input query
                paraphrased_questions = paraphrase_sentence(q)
                print("\nParaphrased Questions :: ")
                
                for i, final_output in enumerate(paraphrased_questions):
                    print(f"{i+1}: {final_output}")

                #need to check if we can pass the paraphrased_questions to the embedder.encde call

                query = embedder.encode(q, convert_to_tensor=True) # we encode our user input 

                from sentence_transformers import util
                search_results = util.semantic_search(query, embeddings, top_k = 3)
                # Display results with enhanced styling
                st.markdown("## 📋 Search Results")
                 
                 # Collect all relevant chunks for LLM processing
                relevant_chunks = []
                for index, result in enumerate(search_results[0]):
                    chunk_text = chunks[result["corpus_id"]].text
                    relevant_chunks.append(f"Result {index+1} (Score: {result['score']:.3f}): {chunk_text}")
                     
                     # Display individual results
                    with st.expander(f"🔍 Result {index+1} (Relevance: {result['score']:.3f})", expanded=index==0):
                        st.write(chunk_text)
                 
                 # Use Ollama LLM to provide a comprehensive answer
                st.markdown("## 🤖 AI Analysis")
                 
                try:
                    from ollama import Client
                    client = Client()
                     
                     # Prepare context for LLM
                    context = "\n\n".join(relevant_chunks)
                    print("line above promt") 
                    prompt = f"""
                    Based on the following patent document search results, provide a comprehensive answer to the user's query: "{q}"
                    
                    Search Results:
                    {context}
                     
                    Please provide:
                    1. A direct answer to the user's query based on the search results
                    2. Key insights from the most relevant results
                    3. How the search results relate to the user's question
                     
                    Answer: """
                    print("line below promt") 
                    with st.spinner("🤖 AI is analyzing the results..."):
                        response = client.generate(model='llama3.1', prompt=prompt)
                        ai_response = response['response']
                     
                     # Display AI response in a nice format
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                border-left: 4px solid #28a745; border-radius: 10px; 
                                padding: 1rem; margin: 1rem 0; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h4 style="margin-top: 0; color: #28a745;">🧠 AI Analysis</h4>
                        <p style="margin-bottom: 0; line-height: 1.6;">{ai_response}</p>
                    </div>
                    """, unsafe_allow_html=True)
                     
                except Exception as e:
                    st.error(f"❌ Error with AI analysis: {e}")
                    st.info("💡 AI analysis unavailable, but you can still view the search results above.")
                print("line below ai response") 
                                         
            except Exception as e:
                st.error(f"❌ Search error: {e}")

def validate_input(text: str):
    """Combine rule-based and AI-based guardrails."""
    issues = check_basic_rules(text)
    check_toxicity_result = check_toxicity(text, threshold=0.5)

    if check_toxicity_result:
        print("Detected Toxicity Categories:")
    for category, score in check_toxicity_result.items():
        print(f"  - {category}: {score:.4f}")
        return False
    else:
        print("No significant toxicity detected.")
        

    if issues:
        print("Input blocked due to:")
        for i in issues:
            print(f" - {i}")
        return False
    else:
        print("Input passed moderation.")
        return True


# Check if running with Streamlit by looking for streamlit in the call stack
import sys
import inspect

def is_running_with_streamlit():
    """Check if the script is being run by Streamlit."""
    for frame_info in inspect.stack():
        if 'streamlit' in frame_info.filename:
            return True
    return False

# If running with Streamlit, run the app directly
if is_running_with_streamlit():
    print("🚀 Running as Streamlit app...")
    

    # If running as regular Python script
if __name__ == "__main__":
    print("🔄 Setting up dataset and embeddings...")
    
    run_search_app()
    
    print("✅ Setup complete!")
    print("🚀 To run the search app, use: uv run streamlit run src/Semantic_Search.py")