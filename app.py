"""
Production-Quality RAG Application with Streamlit
==================================================
A fully local RAG system for PDF Q&A with citations.
No API keys required - runs entirely on CPU.

Architecture:
1. PDF Processing: Extract text with page metadata using PyMuPDF
2. Chunking: Split text into overlapping chunks with page tracking
3. Embeddings: Local sentence-transformers model (BAAI/bge-small-en)
4. Vector Store: FAISS with local persistence
5. Retrieval: Cosine similarity search (top-k)
6. LLM: Local transformers model (TinyLlama)
7. Citations: Page numbers + snippets from retrieved chunks
"""

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

CHUNK_SIZE = 600  # tokens (approx)
CHUNK_OVERLAP = 100  # tokens
TOP_K = 4  # number of chunks to retrieve

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks_metadata" not in st.session_state:
    st.session_state.chunks_metadata = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "llm_pipeline" not in st.session_state:
    st.session_state.llm_pipeline = None


# ============================================================================
# PDF PROCESSING FUNCTIONS
# ============================================================================

def load_pdf(pdf_file) -> List[Dict[str, any]]:
    """
    Extract text from PDF with page metadata.
    
    Args:
        pdf_file: Uploaded PDF file from Streamlit
        
    Returns:
        List of dicts with 'page' and 'text' keys
    """
    try:
        # Read PDF bytes
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        pages_data = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Clean text
            text = clean_text(text)
            
            if text.strip():  # Only add non-empty pages
                pages_data.append({
                    "page": page_num + 1,  # 1-indexed for user display
                    "text": text
                })
        
        pdf_document.close()
        
        if not pages_data:
            raise ValueError("No text extracted from PDF")
            
        return pages_data
        
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and artifacts.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


# ============================================================================
# CHUNKING FUNCTIONS
# ============================================================================

def chunk_text(pages_data: List[Dict], chunk_size: int = CHUNK_SIZE, 
               overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split text into overlapping chunks while preserving page metadata.
    
    Args:
        pages_data: List of page dictionaries with 'page' and 'text'
        chunk_size: Approximate number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk dictionaries with 'text', 'page', and 'chunk_id'
    """
    chunks = []
    chunk_id = 0
    
    for page_data in pages_data:
        page_num = page_data['page']
        text = page_data['text']
        
        # Approximate tokens by splitting on whitespace
        words = text.split()
        
        # Convert token counts to word counts (rough approximation)
        # Typically 1 token ≈ 0.75 words
        words_per_chunk = int(chunk_size * 0.75)
        words_overlap = int(overlap * 0.75)
        
        if len(words) <= words_per_chunk:
            # Page fits in one chunk
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "page": page_num
            })
            chunk_id += 1
        else:
            # Split page into overlapping chunks
            start = 0
            while start < len(words):
                end = start + words_per_chunk
                chunk_words = words[start:end]
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page_num
                })
                chunk_id += 1
                
                # Move start position with overlap
                start += (words_per_chunk - words_overlap)
    
    return chunks


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """
    Load the sentence-transformers embedding model.
    Cached to avoid reloading on every interaction.
    """
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for all chunks.
    
    Args:
        chunks: List of chunk dictionaries
        model: SentenceTransformer model
        
    Returns:
        NumPy array of embeddings (shape: [num_chunks, embedding_dim])
    """
    try:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return np.array([])


# ============================================================================
# VECTOR STORE FUNCTIONS
# ============================================================================

def build_vectorstore(embeddings: np.ndarray, chunks: List[Dict]) -> Optional[faiss.Index]:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: NumPy array of embeddings
        chunks: List of chunk dictionaries (stored separately)
        
    Returns:
        FAISS index or None if error
    """
    try:
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        index.add(embeddings)
        
        return index
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        return None


def save_vectorstore(index: faiss.Index, chunks: List[Dict]):
    """
    Save FAISS index and chunks metadata to disk.
    """
    try:
        faiss.write_index(index, str(STORAGE_DIR / "faiss_index.bin"))
        with open(STORAGE_DIR / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")


def load_vectorstore() -> Tuple[Optional[faiss.Index], List[Dict]]:
    """
    Load FAISS index and chunks metadata from disk.
    
    Returns:
        Tuple of (FAISS index, chunks metadata)
    """
    try:
        index_path = STORAGE_DIR / "faiss_index.bin"
        chunks_path = STORAGE_DIR / "chunks_metadata.pkl"
        
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
        return None, []
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None, []


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_chunks(query: str, index: faiss.Index, chunks: List[Dict], 
                   model: SentenceTransformer, k: int = TOP_K) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    
    Args:
        query: User question
        index: FAISS index
        chunks: List of chunk dictionaries
        model: SentenceTransformer model
        k: Number of chunks to retrieve
        
    Returns:
        List of retrieved chunk dicts with 'text', 'page', 'score'
    """
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, k)
        
        # Gather results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks):  # Valid index
                chunk = chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)
        
        return results
    except Exception as e:
        st.error(f"Error during retrieval: {str(e)}")
        return []


# ============================================================================
# LLM FUNCTIONS
# ============================================================================

@st.cache_resource
def load_llm():
    """
    Load local LLM (TinyLlama) for answer generation.
    Cached to avoid reloading.
    """
    try:
        with st.spinner("Loading local LLM (first time may take a few minutes)..."):
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu"
            )
            
            llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return llm_pipeline
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None


def build_rag_prompt(query: str, retrieved_chunks: List[Dict], 
                    chat_history: List[Dict]) -> str:
    """
    Build RAG prompt with retrieved context and chat history.
    
    Args:
        query: Current user question
        retrieved_chunks: List of retrieved chunks
        chat_history: Previous conversation messages
        
    Returns:
        Formatted prompt string
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Excerpt {i} - Page {chunk['page']}]\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Build chat history (last 3 exchanges to keep prompt manageable)
    history_text = ""
    if chat_history:
        recent_history = chat_history[-6:]  # Last 3 Q&A pairs
        for msg in recent_history:
            role = msg['role'].capitalize()
            history_text += f"{role}: {msg['content']}\n"
    
    # Construct prompt - avoid f-string with special tokens
    system_token_start = "<|system|>"
    system_token_end = "</|system|>"
    user_token_start = "<|user|>"
    user_token_end = "</|user|>"
    assistant_token = "<|assistant|>"
    
    history_section = f"Previous Conversation:\n{history_text}\n\n" if history_text else ""
    
    prompt = f"""{system_token_start}
You are a helpful assistant that answers questions based ONLY on the provided document excerpts.

RULES:
1. Answer ONLY using information from the excerpts below
2. If the answer is not in the excerpts, respond with "Not found in document"
3. ALWAYS cite page numbers in your answer using format: (Page X)
4. Be concise and accurate
5. Do not make up or infer information not present in the excerpts

{history_section}Document Excerpts:
{context}
{system_token_end}

{user_token_start}
{query}
{user_token_end}

{assistant_token}"""
    
    return prompt


def generate_answer(query: str, retrieved_chunks: List[Dict], 
                   llm_pipeline, chat_history: List[Dict]) -> str:
    """
    Generate answer using LLM with RAG prompt.
    
    Args:
        query: User question
        retrieved_chunks: Retrieved context chunks
        llm_pipeline: HuggingFace pipeline
        chat_history: Previous messages
        
    Returns:
        Generated answer string
    """
    try:
        # Handle case with no retrieved chunks
        if not retrieved_chunks:
            return "Not found in document."
        
        # Build prompt
        prompt = build_rag_prompt(query, retrieved_chunks, chat_history)
        
        # Generate
        response = llm_pipeline(prompt, max_new_tokens=300, pad_token_id=llm_pipeline.tokenizer.eos_token_id)
        generated_text = response[0]['generated_text']
        
        # Extract only the assistant's response
        answer = generated_text.split("<|assistant|>")[-1].strip()
        
        # Clean up any remaining special tokens
        answer = answer.replace("<|endoftext|>", "").strip()
        
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Error generating answer. Please try again."


def format_answer_with_citations(answer: str, retrieved_chunks: List[Dict]) -> str:
    """
    Enhance answer with citation snippets if not already present.
    
    Args:
        answer: Generated answer text
        retrieved_chunks: Retrieved chunks for reference
        
    Returns:
        Answer with citations
    """
    # If answer already has citations or is "not found", return as is
    if "(Page" in answer or "not found" in answer.lower():
        return answer
    
    # Otherwise, append citation information
    pages = sorted(set(chunk['page'] for chunk in retrieved_chunks))
    pages_str = ", ".join(str(p) for p in pages)
    
    citation_info = f"\n\n**Sources:** Page(s) {pages_str}"
    
    return answer + citation_info


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF RAG Assistant")
    st.markdown("Upload a PDF and chat with it using local AI (no API keys required)")
    
    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file
            if "current_pdf_name" not in st.session_state or \
               st.session_state.current_pdf_name != uploaded_file.name:
                
                st.session_state.current_pdf_name = uploaded_file.name
                
                with st.spinner("Processing PDF..."):
                    # 1. Load embedding model
                    if st.session_state.embedding_model is None:
                        st.session_state.embedding_model = load_embedding_model()
                    
                    if st.session_state.embedding_model is None:
                        st.error("Failed to load embedding model")
                        return
                    
                    # 2. Extract text from PDF
                    pages_data = load_pdf(uploaded_file)
                    
                    if not pages_data:
                        st.error("Failed to extract text from PDF")
                        return
                    
                    st.success(f"✅ Extracted {len(pages_data)} pages")
                    
                    # 3. Chunk text
                    chunks = chunk_text(pages_data)
                    st.success(f"✅ Created {len(chunks)} chunks")
                    
                    # 4. Generate embeddings
                    embeddings = embed_chunks(chunks, st.session_state.embedding_model)
                    
                    if embeddings.size == 0:
                        st.error("Failed to generate embeddings")
                        return
                    
                    st.success(f"✅ Generated embeddings")
                    
                    # 5. Build vector store
                    vectorstore = build_vectorstore(embeddings, chunks)
                    
                    if vectorstore is None:
                        st.error("Failed to build vector store")
                        return
                    
                    # Save to session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunks_metadata = chunks
                    
                    # Save to disk
                    save_vectorstore(vectorstore, chunks)
                    
                    st.success("✅ Vector store ready!")
                    
                    # Clear previous chat history for new document
                    st.session_state.messages = []
            else:
                st.info(f"📄 **Current document:** {uploaded_file.name}")
                st.info(f"💾 **Chunks indexed:** {len(st.session_state.chunks_metadata)}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.markdown(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")
        st.markdown(f"**LLM Model:** {LLM_MODEL_NAME}")
        st.markdown(f"**Top-K Retrieval:** {TOP_K}")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.vectorstore is None:
        st.info("👈 Please upload a PDF document to get started")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Load LLM if not already loaded
                if st.session_state.llm_pipeline is None:
                    st.session_state.llm_pipeline = load_llm()
                
                if st.session_state.llm_pipeline is None:
                    st.error("Failed to load LLM")
                    return
                
                # Retrieve relevant chunks
                retrieved_chunks = retrieve_chunks(
                    prompt,
                    st.session_state.vectorstore,
                    st.session_state.chunks_metadata,
                    st.session_state.embedding_model,
                    k=TOP_K
                )
                
                # Generate answer
                answer = generate_answer(
                    prompt,
                    retrieved_chunks,
                    st.session_state.llm_pipeline,
                    st.session_state.messages[:-1]  # Exclude current user message
                )
                
                # Format with citations
                final_answer = format_answer_with_citations(answer, retrieved_chunks)
                
                # Display answer
                st.markdown(final_answer)
                
                # Show retrieved chunks in expander
                if retrieved_chunks:
                    with st.expander("📚 View Retrieved Context"):
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            st.markdown(f"**Excerpt {i} (Page {chunk['page']}) - Similarity: {chunk['score']:.3f}**")
                            st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                            st.markdown("---")
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    main()
