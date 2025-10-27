#Trying to use GPU
import torch
import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from query_guadrails import check_toxicity
from query_guadrails import check_basic_rules
from prompt_paraphase import paraphrase_sentence


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
                🔍 College Admissions Search
            </h1>
            <h3 style="margin: 0; color: #2C3E50; font-style: italic; font-size: 1.2rem;">
                *Discover college admission guidance through AI-powered search*
            </h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.selectbox("Choose a page:", ["🔍 Search", "🏗️ Architecture"])
    
    # Check which page to display
    if page == "🏗️ Architecture":
        # Display architecture diagram on main screen
        st.markdown("## 🏗️ System Architecture")
        st.markdown("**Interactive architecture diagram of the College Admissions Search system**")
        
        # Read and display the architecture diagram
        try:
            with open("architecture_diagram.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=1000, scrolling=True)
        except FileNotFoundError:
            st.error("Architecture diagram file not found. Please ensure 'architecture_diagram.html' is in the project root.")
        except Exception as e:
            st.error(f"Error loading architecture diagram: {e}")
    else:
        # Search section with enhanced styling
        st.markdown("## 🔍 AI-Powered Search")
        st.markdown("**Describe what you're looking for and let AI find the perfect answer to your question!**")
        
        # Initialize session state for search query
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        
        # User input with better layout - spread horizontally
        col1,col2 = st.columns([8,2])
        
        with col1:
            q = st.text_area(
                "🎯 Search Query", 
                value=st.session_state.search_query,
                placeholder="e.g., 'How can I prepare for the SAT?'",
                help="Describe the college admission info you're looking for in natural language",
                key="search_input",
                height=100
            ) 

        # Perform search when user enters a query
        if q:
            # Input validation and safety check       
            with st.spinner("Searching..."):
                try:
                    print(f'Query: {q}')
                    #added to check the toxicity in the input query
                    if not validate_input(q):
                        st.markdown("Detected Toxicity in the input query")
                        return False
                    
                    #getting paraphrase questions for the input query
                    #paraphrased_questions = paraphrase_sentence(q)
                    #print("\nParaphrased Questions :: ")
                    
                    #for i, final_output in enumerate(paraphrased_questions):
                    #    print(f"{i+1}: {final_output}")

                    # Use Qdrant search function
                    from search import search_qdrant
                    search_results = search_qdrant(q)
                    
                    from search import search_qdrant_queries
                    search_results_1 = search_qdrant_queries(paraphrased_questions)
                    
                    # Display search results
                    #need to check if we can pass the paraphrased_questions to the embedder.encode call

                    #from sentence_transformers import util
                    #search_results = util.semantic_search(query, embeddings, top_k = 3)
                    # Display results with enhanced styling
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
                        Based on the following college admission document search results, provide a comprehensive answer to the user's query: "{q}"
                        
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