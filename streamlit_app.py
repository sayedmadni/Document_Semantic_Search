import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
from src.Document_Semantic_Search.process_multiple_flies import (
    process_documents, 
    retrieve_context_with_scores, 
    answer_question_with_llm
)

# Page configuration
st.set_page_config(
    page_title="Document Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .similarity-bar {
        background-color: #e1f5fe;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'in_memory_store' not in st.session_state:
    st.session_state.in_memory_store = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Document Semantic Search</h1>', unsafe_allow_html=True)
    st.markdown("Upload PDF documents and ask questions to get AI-powered answers with similarity scores.")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to search through"
        )
        
        # URL input as alternative
        st.subheader("Or enter PDF URLs")
        url_input = st.text_area(
            "Enter PDF URLs (one per line)",
            height=100,
            help="Enter URLs of PDF files to download and process"
        )
        
        # Process button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files or url_input.strip():
                process_documents_ui(uploaded_files, url_input)
            else:
                st.error("Please upload files or enter URLs")
        
        # Display processing status
        if st.session_state.documents_processed:
            st.success(f"✅ Processed {len(st.session_state.processed_documents)} documents")
            st.info(f"📊 {len(st.session_state.in_memory_store)} chunks created")
    
    # Main content area
    if st.session_state.documents_processed:
        # Search interface
        st.header("🔍 Search Documents")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic of the documents?",
                help="Ask any question about the uploaded documents"
            )
        
        with col2:
            k_results = st.number_input(
                "Number of results:",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of relevant chunks to retrieve"
            )
        
        if st.button("🔍 Search", type="primary") and query:
            search_documents(query, k_results)
    else:
        # Welcome message
        st.info("👈 Please upload PDF documents in the sidebar to get started!")
        
        # Demo section
        with st.expander("🚀 How it works"):
            st.markdown("""
            **Document Semantic Search** uses advanced AI to:
            
            1. **Parse & Chunk**: Extract text from PDFs and break it into meaningful chunks
            2. **Embed**: Convert text chunks into vector embeddings for similarity search
            3. **Search**: Find the most relevant chunks based on your query
            4. **Answer**: Generate AI-powered answers using the retrieved context
            
            **Features:**
            - 📊 **Similarity Scores**: See how relevant each chunk is to your query
            - 📈 **Visualizations**: Interactive charts showing relevance scores
            - 🎯 **Document Attribution**: Know which document each answer comes from
            - 🤖 **AI Answers**: Get comprehensive answers powered by Ollama/LLaMA
            """)

def process_documents_ui(uploaded_files, url_input):
    """Process uploaded documents and update session state"""
    with st.spinner("Processing documents..."):
        try:
            # Prepare document sources
            document_sources = []
            
            # Handle uploaded files
            if uploaded_files:
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    document_sources.append(temp_path)
            
            # Handle URL input
            if url_input.strip():
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                document_sources.extend(urls)
            
            # Process documents
            in_memory_store, embedder = asyncio.run(process_documents(document_sources))
            
            # Update session state
            st.session_state.in_memory_store = in_memory_store
            st.session_state.embedder = embedder
            st.session_state.documents_processed = True
            st.session_state.processed_documents = document_sources
            
            # Clean up temp files
            if uploaded_files:
                import shutil
                shutil.rmtree(temp_dir)
            
            st.success("Documents processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

def search_documents(query, k_results):
    """Search documents and display results"""
    with st.spinner("Searching documents..."):
        try:
            # Retrieve relevant chunks with scores
            results = retrieve_context_with_scores(
                query, 
                st.session_state.in_memory_store, 
                st.session_state.embedder, 
                k=k_results
            )
            
            if not results:
                st.warning("No relevant results found.")
                return
            
            # Display results
            st.subheader("📊 Search Results")
            
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Results Found", len(results))
            with col2:
                avg_score = sum(r['similarity_score'] for r in results) / len(results)
                st.metric("Avg Similarity", f"{avg_score:.3f}")
            with col3:
                max_score = max(r['similarity_score'] for r in results)
                st.metric("Highest Score", f"{max_score:.3f}")
            
            # Similarity scores visualization
            st.subheader("📈 Relevance Scores")
            
            # Create DataFrame for visualization
            df_scores = pd.DataFrame([
                {
                    'Document': result['document_name'],
                    'Similarity Score': result['similarity_score'],
                    'Relevance %': result['relevance_percentage']
                }
                for result in results
            ])
            
            # Bar chart
            fig = px.bar(
                df_scores, 
                x='Document', 
                y='Similarity Score',
                title='Similarity Scores by Document',
                color='Similarity Score',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.subheader("📄 Detailed Results")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}: {result['document_name']} (Score: {result['similarity_score']:.3f})"):
                    # Similarity score visualization
                    score_pct = result['relevance_percentage']
                    st.markdown(f"""
                    <div class="similarity-bar">
                        <strong>Relevance Score:</strong> {result['similarity_score']:.4f} 
                        ({score_pct:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for visual representation
                    st.progress(score_pct / 100)
                    
                    # Document content
                    st.markdown("**Content:**")
                    st.text_area(
                        "Chunk content:",
                        value=result['text'],
                        height=150,
                        key=f"content_{i}",
                        disabled=True
                    )
            
            # Generate AI answer
            st.subheader("🤖 AI Answer")
            
            if st.button("Generate AI Answer", type="secondary"):
                with st.spinner("Generating AI answer..."):
                    # Prepare context for LLM
                    context = ""
                    for result in results:
                        context += f"Document: {result['document_name']}\nContent: {result['text']}\n\n"
                    
                    # Generate answer
                    answer = answer_question_with_llm(query, context)
                    
                    st.markdown("**AI Response:**")
                    st.markdown(answer)
                    
                    # Show source documents
                    with st.expander("📚 Sources"):
                        for result in results:
                            st.markdown(f"**{result['document_name']}** (Score: {result['similarity_score']:.3f})")
                            st.text(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
                            st.markdown("---")
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")

if __name__ == "__main__":
    main()
