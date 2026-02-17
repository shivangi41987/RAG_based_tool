#  PDF RAG Assistant - Production Quality

A fully local, production-quality Retrieval-Augmented Generation (RAG) system for PDF Q&A with citations. **No API keys required** - runs entirely on your CPU.

##  Features

- **100% Local**: No OpenAI or external APIs - complete offline functionality
- **PDF Upload**: Upload any PDF document through Streamlit interface
- **Smart Chunking**: Overlapping chunks (600 tokens) with page metadata preservation
- **Semantic Search**: FAISS vector store with cosine similarity retrieval
- **Local Embeddings**: Uses `BAAI/bge-small-en` from sentence-transformers
- **Local LLM**: TinyLlama-1.1B for answer generation
- **Citations**: Every answer includes page numbers and source snippets
- **Chat History**: Persistent conversation context for follow-up questions
- **Error Handling**: Robust handling of edge cases and empty results
- **Production Ready**: Clean, modular code with comprehensive comments

##  Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will download models (~2GB):
- BAAI/bge-small-en (133MB) - Embedding model
- TinyLlama-1.1B-Chat-v1.0 (~2.2GB) - Language model

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Usage

1. **Upload PDF**: Click "Browse files" in the sidebar
2. **Wait for Processing**: PDF will be chunked and indexed (shows progress)
3. **Ask Questions**: Type your question in the chat input
4. **View Citations**: Expand "View Retrieved Context" to see source excerpts

##  Expected Test Cases

### 1. Direct Fact Retrieval
**Q**: *What is the candidate's total experience?*  
**Expected**: `The candidate has 2 years of experience. (Page 1)`

###  2. Skill Lookup
**Q**: *What programming languages does the candidate know?*  
**Expected**: `The candidate knows Python, SQL, and C++. (Page 1)`

###  3. Work Experience Detail
**Q**: *What was the candidate's role at ABC Company?*  
**Expected**: `The candidate worked as a Data Scientist at ABC Company. (Page 2)`

###  4. Responsibility Extraction
**Q**: *What were the key responsibilities in the last job?*  
**Expected**: `The responsibilities included building ML models, data preprocessing, and dashboard development. (Page 2)`

###  5. Project-Specific Question
**Q**: *What technologies were used in the RAG project?*  
**Expected**: `The project used FAISS, SentenceTransformers, and a local LLM. (Page 3)`

###  6. Multi-Chunk Reasoning
**Q**: *Does the candidate have experience in both machine learning and data engineering?*  
**Expected**: `Yes, the candidate has experience in machine learning and data engineering. (Pages 1 and 2)`

###  7. Not Found Case (Critical)
**Q**: *What is the candidate's GPA?*  
**Expected**: `Not found in document.`

###  8. Follow-Up Question
**First Q**: *What tools does she use?*  
**Follow-up**: *Does she use PySpark?*  
**Expected**: System understands "she" from conversation history

###  9. Edge Case - Ambiguous Question
**Q**: *When did she start?*  
**Expected**: Answers based on most recent or clarifies if ambiguous

###  10. Long Answer Question
**Q**: *Summarize the candidate's experience.*  
**Expected**: Concise summary from context with citations

##  Architecture

### Pipeline Flow

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              ↓
User Query → Query Embedding → Similarity Search → Top-K Chunks
                                                              ↓
                        RAG Prompt ← Context + Chat History
                                                              ↓
                                         Local LLM → Answer + Citations
```

### Components

1. **PDF Processing** (`load_pdf`, `clean_text`)
   - PyMuPDF for text extraction
   - Page metadata preservation
   - Text cleaning and normalization

2. **Chunking** (`chunk_text`)
   - 600 tokens per chunk (approx.)
   - 100 token overlap
   - Page tracking for citations

3. **Embeddings** (`embed_chunks`, `load_embedding_model`)
   - sentence-transformers: BAAI/bge-small-en
   - 384-dimensional embeddings
   - Cached model loading

4. **Vector Store** (`build_vectorstore`)
   - FAISS IndexFlatIP (cosine similarity)
   - L2 normalization
   - Local persistence (./storage/)

5. **Retrieval** (`retrieve_chunks`)
   - Top-4 chunks by default
   - Similarity scoring
   - Metadata preservation

6. **LLM** (`generate_answer`, `load_llm`)
   - TinyLlama-1.1B-Chat-v1.0
   - Structured prompt with RULES
   - Chat history integration

7. **Citations** (`format_answer_with_citations`)
   - Page numbers from retrieved chunks
   - Snippet display in expander
   - Automatic citation formatting

##  Project Structure

```
rag_assignment/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── storage/              # Auto-created for persistence
    ├── faiss_index.bin   # Vector store index
    └── chunks_metadata.pkl  # Chunk metadata
```

## ⚙️ Configuration

Edit these constants in `app.py` to customize behavior:

```python
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"  # Embedding model
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # LLM
CHUNK_SIZE = 600  # Tokens per chunk
CHUNK_OVERLAP = 100  # Overlap tokens
TOP_K = 4  # Retrieved chunks
```

### Alternative Models

**Embeddings:**
- `all-MiniLM-L6-v2` (384 dim, faster)
- `BAAI/bge-base-en` (768 dim, more accurate)

**LLM:**
- `microsoft/phi-2` (2.7B, better quality, slower)
- `google/flan-t5-large` (780M, instruction-tuned)

##  Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce CHUNK_SIZE or use smaller models

### Issue: Slow Response Time
**Solution**: Reduce TOP_K or max_new_tokens in `generate_answer()`

### Issue: Model Download Fails
**Solution**: Check internet connection; models are downloaded on first run

### Issue: "Not found in document" for valid questions
**Solution**: 
- Try different phrasing
- Check if PDF text extracted correctly
- Increase TOP_K for broader retrieval

##  Privacy & Security

- **100% Local**: No data leaves your machine
- **No API Keys**: No external service dependencies
- **Offline Capable**: Works without internet after models are downloaded

##  Performance

**Typical Performance (CPU):**
- PDF Processing: ~2-5 seconds per page
- Embedding Generation: ~0.5 seconds per chunk
- Query Response: ~10-30 seconds (LLM generation)

**Hardware Requirements:**
- RAM: 8GB minimum (16GB recommended)
- Disk Space: 5GB for models and data
- CPU: Any modern multi-core processor

## Contributing

Suggestions for improvement:
- Add support for multiple PDFs
- Implement semantic caching for faster responses
- Add re-ranking for better retrieval
- Support for tables and images

## License

Open source - feel free to modify and adapt for your needs.

##  Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web interface
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [sentence-transformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Transformers](https://huggingface.co/docs/transformers/) - LLM inference
