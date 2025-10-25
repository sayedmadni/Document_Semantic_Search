# Document Semantic Search - Streamlit Interface

A modern web interface for semantic document search with AI-powered answers and detailed relevance scoring.

## Features

🔍 **Semantic Search**: Find relevant content using vector similarity search
📊 **Relevance Scoring**: See detailed similarity scores and relevance percentages
📈 **Visualizations**: Interactive charts showing document relevance
🤖 **AI Answers**: Get comprehensive answers powered by Ollama/LLaMA
📁 **Multiple Formats**: Support for PDF uploads and URL downloads
🎯 **Document Attribution**: Know which document each answer comes from

## Quick Start

### 1. Install Dependencies

```bash
# Install the project dependencies
pip install -e .
```

### 2. Run the Streamlit App

```bash
# Option 1: Using the launcher script
python run_streamlit.py

# Option 2: Direct streamlit command
streamlit run streamlit_app.py
```

### 3. Open in Browser

The app will automatically open in your browser at `http://localhost:8501`

## How to Use

### Step 1: Upload Documents
- **Upload PDFs**: Use the file uploader in the sidebar
- **Enter URLs**: Paste PDF URLs (one per line) in the text area
- Click "🔄 Process Documents" to parse and embed the documents

### Step 2: Search
- Enter your question in the search box
- Adjust the number of results (1-10)
- Click "🔍 Search" to find relevant chunks

### Step 3: View Results
- **Similarity Scores**: See relevance scores for each result
- **Visualizations**: Interactive bar charts showing document relevance
- **Detailed Results**: Expand each result to see the full content
- **AI Answers**: Generate comprehensive answers using the retrieved context

## Understanding the Scores

### Similarity Score
- Range: 0.0 to 1.0
- Higher values indicate better semantic match
- Based on cosine similarity between query and document embeddings

### Relevance Percentage
- Range: 0% to 100%
- Percentage representation of the similarity score
- Easier to interpret than raw similarity values

### Visual Indicators
- **Progress bars**: Visual representation of relevance
- **Color coding**: Blue gradient from light to dark based on relevance
- **Bar charts**: Compare relevance across multiple documents

## Technical Details

### Architecture
- **Document Processing**: Uses Docling for PDF parsing and Chonkie for chunking
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2) for vector representations
- **Search**: Cosine similarity for semantic matching
- **AI**: Ollama with LLaMA 3.1 for answer generation

### Performance
- **In-memory storage**: Fast retrieval without database overhead
- **Async processing**: Efficient document downloading and processing
- **Batch embeddings**: Optimized vector generation

## Troubleshooting

### Common Issues

1. **"No module named 'streamlit'"**
   ```bash
   pip install streamlit
   ```

2. **"Ollama not found"**
   - Install Ollama: https://ollama.ai/
   - Pull LLaMA model: `ollama pull llama3.1`

3. **PDF processing errors**
   - Ensure PDFs are not password-protected
   - Check file format compatibility

4. **Memory issues with large documents**
   - Process fewer documents at once
   - Consider using smaller chunk sizes

### Performance Tips

- **Batch processing**: Upload multiple documents together
- **Query optimization**: Use specific, well-formed questions
- **Result tuning**: Adjust the number of results based on your needs

## Customization

### Modifying the Interface
- Edit `streamlit_app.py` to customize the UI
- Modify CSS styles in the `st.markdown()` sections
- Add new visualizations using Plotly

### Changing Models
- Update the embedding model in `process_multiple_files.py`
- Modify the LLM model in the `answer_question_with_llm` function

### Adding Features
- Implement document metadata extraction
- Add support for more file formats
- Include document summarization capabilities

## API Reference

### Key Functions

- `process_documents()`: Parse and embed multiple documents
- `retrieve_context_with_scores()`: Search with similarity scores
- `answer_question_with_llm()`: Generate AI answers

### Session State Variables

- `documents_processed`: Boolean indicating if documents are loaded
- `in_memory_store`: List of document chunks with embeddings
- `embedder`: SentenceTransformer embedding function
- `processed_documents`: List of processed document sources

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Verify that Ollama is running and accessible

## License

This project follows the same license as the parent Document Semantic Search project.
