# Quick Start Guide for PDF RAG Assistant

## Step 1: Install Dependencies

Open PowerShell or Command Prompt in this directory and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** The first installation may take 5-10 minutes depending on your internet speed.

## Step 2: Run the Application

```powershell
# Make sure virtual environment is activated
streamlit run app.py
```

The application will:
1. Automatically download required models (~2GB total) on first run
2. Open in your default browser at http://localhost:8501

## Step 3: Test the Application

### Create a Sample Resume PDF for Testing

Create a simple test PDF with content like:

```
JOHN DOE
Data Scientist | 2 Years Experience

SKILLS
- Python, SQL, C++
- Machine Learning, Data Engineering
- FAISS, PySpark, TensorFlow

EXPERIENCE

ABC Company - Data Scientist (2023-Present)
- Built ML models for customer segmentation
- Data preprocessing and ETL pipelines
- Dashboard development using Streamlit

PROJECTS

RAG Document Assistant
- Built a retrieval-augmented generation system
- Used FAISS, SentenceTransformers, and local LLM
- Implemented citation tracking and context chunking

EDUCATION

BS Computer Science
University Name (2021)
```

### Test Questions to Try

1. "What is the candidate's total experience?"
2. "What programming languages does the candidate know?"
3. "What was the candidate's role at ABC Company?"
4. "What were the key responsibilities in the last job?"
5. "What technologies were used in the RAG project?"
6. "Does the candidate have experience in both machine learning and data engineering?"
7. "What is the candidate's GPA?" (Should return "Not found")
8. "What tools does she use?" then "Does she use PySpark?" (Follow-up test)

## Troubleshooting

### Virtual Environment Issues

If `.\venv\Scripts\activate` doesn't work, try:
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
.\venv\Scripts\activate
```

### Port Already in Use

If port 8501 is in use:
```powershell
streamlit run app.py --server.port 8502
```

### Module Import Errors

Make sure virtual environment is activated (you should see `(venv)` in your prompt):
```powershell
# Check if packages are installed
pip list | findstr streamlit
```

### Out of Memory Errors

Edit `app.py` and reduce these values:
- `CHUNK_SIZE = 400` (line 35)
- `TOP_K = 3` (line 37)
- `max_new_tokens=200` (in `load_llm()` function)

## Performance Tips

### First Run
- Model download: ~5 minutes (one-time)
- App startup: ~30-60 seconds (loading models)

### Subsequent Runs
- App startup: ~10-20 seconds
- PDF processing: ~2-5 seconds per page
- Query response: ~10-30 seconds

### Speed Optimization
1. Reduce `TOP_K` from 4 to 3
2. Reduce `max_new_tokens` from 300 to 200
3. Use smaller embedding model: `all-MiniLM-L6-v2`

## Project Structure After Setup

```
rag_assignment/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── README.md                # Documentation
├── QUICKSTART.md            # This file
├── .gitignore               # Git ignore rules
├── venv/                    # Virtual environment (created)
└── storage/                 # Auto-created on first PDF upload
    ├── faiss_index.bin      # Vector database
    └── chunks_metadata.pkl  # Chunk information
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Run the application  
3. ✅ Upload a test PDF
4. ✅ Ask questions and verify citations
5. 🎉 Enjoy your local RAG system!

## Need Help?

- Check the main [README.md](README.md) for detailed architecture
- Review error messages in the Streamlit UI
- Ensure you have at least 8GB RAM available
- Make sure models downloaded successfully (check console output)

Happy chatting with your PDFs! 📚🤖
