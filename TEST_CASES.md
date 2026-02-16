# Test Cases & Expected Behavior

This document outlines all test cases for verifying the PDF RAG Assistant application.

## Test PDF Content

Create a test PDF (or Word document exported as PDF) with the following content:

---

**PAGE 1**

```
JANE SMITH
Data Scientist & ML Engineer

PROFESSIONAL SUMMARY
Experienced Data Scientist with 2 years of experience in machine learning,
data engineering, and analytics. Proven track record of delivering 
production-ready ML solutions.

TECHNICAL SKILLS
Programming Languages: Python, SQL, C++
Machine Learning: TensorFlow, PyTorch, Scikit-learn
Data Engineering: PySpark, Airflow, Kafka
Databases: PostgreSQL, MongoDB, Redis
Tools: Git, Docker, Kubernetes
```

**PAGE 2**

```
WORK EXPERIENCE

ABC Company | Data Scientist | 2023 - Present
• Built ML models for customer segmentation and churn prediction
• Implemented data preprocessing and ETL pipelines
• Dashboard development using Streamlit and Plotly
• Collaborated with cross-functional teams on data strategy
• Reduced model inference time by 40% through optimization

XYZ Corp | Junior Data Analyst | 2022 - 2023
• Performed exploratory data analysis on customer datasets
• Created SQL queries for business intelligence reports
• Automated reporting workflows using Python scripts
```

**PAGE 3**

```
PROJECTS

RAG Document Assistant (2024)
• Built a retrieval-augmented generation system for PDF Q&A
• Technologies: FAISS, SentenceTransformers, local LLM
• Implemented semantic chunking and citation tracking
• Deployed using Streamlit for user-friendly interface

E-commerce Recommendation Engine (2023)
• Developed collaborative filtering recommendation system
• Used PySpark for distributed processing of 10M+ records
• Achieved 25% improvement in click-through rate

EDUCATION

Bachelor of Science in Computer Science
State University | 2018 - 2022
Relevant Coursework: Machine Learning, Databases, Algorithms
```

---

## Test Scenarios

### ✅ Test 1: Direct Fact Retrieval

**Question:**
```
What is the candidate's total experience?
```

**Expected Behavior:**
- Should retrieve chunk from Page 1
- Should mention "2 years of experience"
- Should cite (Page 1)

**Example Answer:**
```
The candidate has 2 years of experience. (Page 1)
```

**Verification:**
- ✅ Answer is factually correct
- ✅ Page citation is present
- ✅ No hallucinated information

---

### ✅ Test 2: Skill Lookup

**Question:**
```
What programming languages does the candidate know?
```

**Expected Behavior:**
- Should retrieve from Page 1 (Technical Skills section)
- Should list Python, SQL, C++
- Should cite (Page 1)

**Example Answer:**
```
The candidate knows Python, SQL, and C++. (Page 1)
```

**Verification:**
- ✅ All three languages mentioned
- ✅ No additional languages added
- ✅ Page citation correct

---

### ✅ Test 3: Work Experience Detail

**Question:**
```
What was the candidate's role at ABC Company?
```

**Expected Behavior:**
- Should retrieve from Page 2 (Work Experience section)
- Should identify "Data Scientist"
- Should cite (Page 2)

**Example Answer:**
```
The candidate worked as a Data Scientist at ABC Company. (Page 2)
```

**Verification:**
- ✅ Role correctly identified
- ✅ Company name correctly matched
- ✅ Page citation accurate

---

### ✅ Test 4: Responsibility Extraction

**Question:**
```
What were the key responsibilities in the last job?
```

**Expected Behavior:**
- Should retrieve from Page 2 (ABC Company section)
- Should list main responsibilities
- Should cite (Page 2)

**Example Answer:**
```
The key responsibilities included building ML models for customer segmentation 
and churn prediction, implementing data preprocessing and ETL pipelines, and 
dashboard development using Streamlit and Plotly. (Page 2)
```

**Verification:**
- ✅ Multiple responsibilities mentioned
- ✅ Specific details from context
- ✅ No made-up responsibilities

---

### ✅ Test 5: Project-Specific Question

**Question:**
```
What technologies were used in the RAG project?
```

**Expected Behavior:**
- Should retrieve from Page 3 (Projects section)
- Should list FAISS, SentenceTransformers, local LLM, Streamlit
- Should cite (Page 3)

**Example Answer:**
```
The RAG project used FAISS, SentenceTransformers, and a local LLM. 
It was deployed using Streamlit. (Page 3)
```

**Verification:**
- ✅ Technologies accurately listed
- ✅ Specific to RAG project (not other projects)
- ✅ Page citation correct

---

### ✅ Test 6: Multi-Chunk Reasoning

**Question:**
```
Does the candidate have experience in both machine learning and data engineering?
```

**Expected Behavior:**
- Should retrieve from multiple pages (1, 2, and/or 3)
- Should confirm both experiences
- Should cite multiple pages if applicable

**Example Answer:**
```
Yes, the candidate has experience in both machine learning and data engineering. 
The technical skills include ML tools like TensorFlow and PyTorch, as well as 
data engineering tools like PySpark and Airflow. (Pages 1 and 2)
```

**Verification:**
- ✅ Both areas confirmed
- ✅ Evidence from context
- ✅ Multiple page citations

---

### ✅ Test 7: Not Found Case (Critical - Hallucination Check)

**Question:**
```
What is the candidate's GPA?
```

**Expected Behavior:**
- GPA is NOT mentioned in the test PDF
- Should return "Not found in document"
- Should NOT make up a GPA

**Example Answer:**
```
Not found in document.
```

**Verification:**
- ✅ Does not hallucinate a GPA
- ✅ Explicitly states information not found
- ✅ No guessing or inference

**CRITICAL:** This test verifies hallucination control. If the system invents a GPA, it FAILS.

---

### ✅ Test 8: Follow-Up Question (Chat History)

**First Question:**
```
What tools does she use?
```

**Expected Answer:**
```
She uses tools including TensorFlow, PyTorch, Scikit-learn, PySpark, Airflow,
Kafka, Git, Docker, and Kubernetes. (Page 1)
```

**Follow-Up Question:**
```
Does she use PySpark?
```

**Expected Behavior:**
- Should understand "she" refers to Jane Smith (candidate)
- Should use chat history for context
- Should find PySpark in Page 1 or 2

**Expected Answer:**
```
Yes, she uses PySpark. It's listed in her data engineering skills. (Page 1)
```

**Verification:**
- ✅ Resolves "she" from conversation context
- ✅ Correctly identifies PySpark
- ✅ Maintains conversational flow

---

### ✅ Test 9: Ambiguous Question

**Question:**
```
When did she start?
```

**Expected Behavior:**
- Multiple start dates exist (ABC Company 2023, XYZ Corp 2022, University 2018)
- Should provide the most recent professional start date
- Or clarify if ambiguous

**Example Answer:**
```
She started at ABC Company in 2023. (Page 2)
```

**Alternative (Better):**
```
She started at ABC Company in 2023 and at XYZ Corp in 2022. (Page 2)
```

**Verification:**
- ✅ Provides at least one start date
- ✅ Citations included
- ✅ Doesn't make up dates

---

### ✅ Test 10: Long Answer Question

**Question:**
```
Summarize the candidate's experience.
```

**Expected Behavior:**
- Should synthesize information from multiple pages
- Should be concise but comprehensive
- Should cite sources
- Should NOT add information not in document

**Example Answer:**
```
Jane Smith is a Data Scientist with 2 years of experience in machine learning 
and data engineering. She currently works at ABC Company where she builds ML 
models, implements ETL pipelines, and develops dashboards. Her technical skills 
include Python, SQL, C++, TensorFlow, PyTorch, and PySpark. She has completed 
projects including a RAG document assistant and an e-commerce recommendation 
engine. She holds a BS in Computer Science from State University. 
(Pages 1, 2, and 3)
```

**Verification:**
- ✅ Comprehensive summary
- ✅ All information from document
- ✅ No hallucinated details
- ✅ Multiple page citations
- ✅ Concise and well-structured

---

## Edge Cases to Test

### Empty Query
**Input:** `""` (empty string)  
**Expected:** Should handle gracefully, possibly ask for input

### Very Long Query
**Input:** 500+ word question  
**Expected:** Should still process, may truncate if needed

### Irrelevant Query
**Question:** `What is the capital of France?`  
**Expected:** `Not found in document.`

### Repeated Questions
**Input:** Ask same question twice  
**Expected:** Should provide consistent answers

### Special Characters
**Question:** `What's the candidate's role @ ABC Company?`  
**Expected:** Should handle special characters gracefully

---

## Success Criteria

The application PASSES if:
1. ✅ All 10 main test cases return correct answers
2. ✅ Test 7 (hallucination check) correctly returns "Not found"
3. ✅ Citations are present in all answers
4. ✅ Page numbers are accurate
5. ✅ No crashes or errors during testing
6. ✅ Follow-up questions maintain context
7. ✅ Retrieved context snippets are viewable in expander

The application FAILS if:
1. ❌ Hallucinates information not in document
2. ❌ Missing citations
3. ❌ Incorrect page numbers
4. ❌ Crashes on valid input
5. ❌ Cannot handle follow-up questions

---

## Testing Checklist

- [ ] Install dependencies successfully
- [ ] Application starts without errors
- [ ] Can upload PDF file
- [ ] PDF is processed and indexed
- [ ] Can ask first question
- [ ] Answer includes citation
- [ ] Can view retrieved context
- [ ] Test all 10 scenarios
- [ ] Test 7 returns "Not found"
- [ ] Follow-up questions work
- [ ] Clear chat history works
- [ ] Can upload new PDF
- [ ] New PDF clears old index
- [ ] Performance acceptable (<30s per query)

---

## Troubleshooting Test Failures

### If citations are missing:
- Check `format_answer_with_citations()` function
- Verify retrieved_chunks have page metadata

### If hallucinations occur:
- Review RAG prompt construction
- Strengthen "ONLY from context" instruction
- Reduce LLM temperature

### If retrieval fails:
- Verify embeddings are generated
- Check FAISS index is built correctly
- Try increasing TOP_K

### If follow-ups fail:
- Verify chat_history is passed to prompt
- Check session_state.messages is persisting

---

## Performance Benchmarks

**Target Performance:**
- PDF Upload & Processing: <5 seconds per page
- First Query: <30 seconds (model loading)
- Subsequent Queries: <15 seconds
- Memory Usage: <4GB RAM

**If performance is poor:**
- Reduce CHUNK_SIZE
- Reduce TOP_K
- Reduce max_new_tokens
- Consider using smaller models

---

Good luck with testing! 🚀
