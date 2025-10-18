#Trying to use GPU
import torch
import sys
from pathlib import Path
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def run_search_app():
    """Launch the Streamlit search interface."""
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
        def validate_query(query): # This is where Sangeetha will do the Ensamble Prompting / Query Guardrails item 11 on google doc
            pass
        
        with st.spinner("Searching..."):
            try:
                print(f'Query: {q}')
                
                # Use Qdrant search function
                from search import search_qdrant
                search_results = search_qdrant(q)
                
                # Display search results
                st.markdown("## 📋 Search Results")
                
                # Collect all relevant chunks for LLM processing
                relevant_chunks = []
                for index, result in enumerate(search_results):
                    relevant_chunks.append(result)
                    
                    # Display individual results
                    with st.expander(f"🔍 Result {index+1}", expanded=index==0):
                        # Extract just the text part from the result
                        result_text = result.split(": ", 1)[1] if ": " in result else result
                        st.write(result_text)

                 # Use Ollama LLM to provide a comprehensive answer
                st.markdown("## 🤖 AI Analysis")
                 
                try:
                    from ollama import Client
                    client = Client()
                     
                     # Prepare context for LLM
                    context = "\n\n".join(relevant_chunks)

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
                                         
            except Exception as e:
                st.error(f"❌ Search error: {e}")

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