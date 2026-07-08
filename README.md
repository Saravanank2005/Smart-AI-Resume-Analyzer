

# Smart AI Resume Analyzer & Career Coach

A semantic ATS recruiter and resume optimization consultant powered by the Hugging Face Serverless Inference API, a local ChromaDB vector database, and Gradio.

## 🚀 Live Demo
You can access the live application here: [Hugging Face Space](https://huggingface.co/spaces/Sarn007/AI_Resume_Analyzer)

## 🌟 Key Features
- **ATS Compatibility Scoring**: Ranks and matches resumes against specific job descriptions.
- **Semantic RAG Ingestion**: Queries a localized **ChromaDB** database filled with curated job profile parameters to evaluate role requirements.
- **Skill Gap Analysis**: Identifies matching and missing skills represented as dynamic visual badges.
- **Collapsible Career Coach Chatbot**: Offers inline recommendations, interview questions, and STAR-method resume bullet rewrites.

## 🛠️ Tech Stack
- **Frontend**: Gradio (Python)
- **Database**: ChromaDB (Vector Database)
- **Embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Inference Engine**: Hugging Face Inference API (`Qwen/Qwen2.5-72B-Instruct`)
- **CI/CD**: GitHub Actions (automated sync workflow)
