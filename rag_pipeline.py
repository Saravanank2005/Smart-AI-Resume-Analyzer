from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import chromadb
import os
import json
import re

# Load environment variables
load_dotenv()

# =========================
# FIND FOLDERS ROBUSTLY
# =========================

def find_folder(folder_name):
    # Try current directory, then ChatbotAI/, then script's directory/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        folder_name,
        os.path.join("ChatbotAI", folder_name),
        os.path.join(script_dir, folder_name),
        os.path.join(script_dir, "ChatbotAI", folder_name)
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return c
    return None

# =========================
# LOAD DOCUMENTS
# =========================

documents = []

notes_path = find_folder("notes")
if notes_path:
    for file in os.listdir(notes_path):
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(notes_path, file), encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading note {file}: {e}")

jk_path = find_folder("job_knowledge")
if jk_path:
    for file in os.listdir(jk_path):
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(jk_path, file), encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading job profile {file}: {e}")

# =========================
# SPLIT DOCUMENTS
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(documents) if documents else []

# =========================
# EMBEDDINGS & CHROMADB
# =========================

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [doc.page_content for doc in split_docs]

if texts:
    embeddings = embedding_model.encode(texts)
    print("Embeddings Created")
else:
    embeddings = []

client = chromadb.Client()
collection = client.get_or_create_collection(name="echomind")

if texts and len(embeddings) > 0:
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            ids=[str(i)]
        )
    print(f"Stored {len(texts)} chunks in ChromaDB")

# =========================
# SEARCH FUNCTION (Local RAG)
# =========================

def search_notes(query, top_k=3):
    if not texts:
        return []
    try:
        query_embedding = embedding_model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results['documents'][0] if results and 'documents' in results and results['documents'] else []
    except Exception as e:
        print(f"RAG search error (ignored): {e}")
        return []

# =========================
# FILE PARSERS FOR RESUMES
# =========================

def extract_text_from_pdf(file_path):
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading Word document: {e}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        return f"Error reading text file: {e}"

def extract_resume_text(file_path):
    if not file_path:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        return extract_text_from_txt(file_path)

# =========================
# ROBUST JSON PARSER
# =========================

def extract_json(text):
    text = text.strip()
    # Try parsing directly
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Try locating the JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except Exception as e:
            # Clean common formatting issues like unescaped newlines in quotes
            try:
                def replace_nl(match):
                    return match.group(0).replace('\n', '\\n')
                cleaned = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', replace_nl, json_str)
                return json.loads(cleaned)
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON: {e2}. Original error: {e}")
    raise ValueError("No JSON object found in response.")

# =========================
# ATS RESUME ANALYSIS (HF LLM)
# =========================

def analyze_resume_ats(resume_text, required_skills="", target_role="", model_name="Qwen/Qwen2.5-72B-Instruct", hf_token=None):
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token or hf_token.strip() in ("", "YOUR_HF_TOKEN"):
        return {
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "keyword_analysis": "Error: Hugging Face API token is missing.",
            "formatting_issues": [],
            "recommendations": ["Please configure HF_TOKEN in your environment variables or paste it in the Settings panel."],
            "interview_questions": ["Hugging Face Token is missing. Unable to generate interview questions."]
        }

    # 1. RAG step: Retrieve relevant job-skill knowledge based on target role
    retrieved_context = ""
    if target_role:
        retrieved_chunks = search_notes(target_role, top_k=3)
        if retrieved_chunks:
            retrieved_context = "\n".join(retrieved_chunks)

    system_instruction = """You are an expert ATS (Applicant Tracking System) recruiter and resume optimization consultant.
Analyze the user's resume text against the target job role and required skills. Use the retrieved job-skill knowledge as reference guidelines.
Identify the match score, matching skills, missing skills, keyword gaps, and structural/formatting issues.
Also, generate a list of 5 role-specific interview questions (a mix of technical and behavioral) tailored to the resume and target role.

Return your response strictly as a JSON object with the following structure:
{
  "match_score": integer (0 to 100),
  "matching_skills": list of strings,
  "missing_skills": list of strings,
  "keyword_analysis": string (brief summary of keyword match quality),
  "formatting_issues": list of strings (structural flaws or formatting alerts),
  "recommendations": list of strings (actionable edit suggestions to improve the resume match),
  "interview_questions": list of strings (5 tailored, highly specific interview questions)
}
Do not return any markdown code block wrap, HTML, or conversational filler outside the JSON."""

    prompt = f"""### Target Job Role:
{target_role if target_role else "Not Specified"}

### Extra Required Skills / Job Description:
{required_skills if required_skills else "Not Specified"}

### Retrieved Job-Skill Knowledge (RAG Reference):
{retrieved_context if retrieved_context else "None"}

### Resume Text:
{resume_text}
"""

    try:
        client = InferenceClient(model=model_name, token=hf_token)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.1
        )
        result_text = response.choices[0].message.content
        analysis = extract_json(result_text)
        return analysis
    except Exception as e:
        # Try a direct requests post as a fallback
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": 0.1
                }
            }
            res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                result_text = res.json()[0]['generated_text'] if isinstance(res.json(), list) else res.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                analysis = extract_json(result_text)
                return analysis
            else:
                raise Exception(f"HTTP Status {res.status_code}: {res.text}")
        except Exception as fallback_err:
            return {
                "match_score": 0,
                "matching_skills": [],
                "missing_skills": [],
                "keyword_analysis": f"Error during analysis: {e}. Fallback error: {fallback_err}",
                "formatting_issues": [],
                "recommendations": [f"An error occurred while communicating with Hugging Face: {e}"],
                "interview_questions": [f"Failed to generate questions: {e}"]
            }

# =========================
# GLOBAL CHATBOT (HF LLM)
# =========================

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])
                elif "content" in part:
                    text_parts.append(extract_text(part["content"]))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    elif isinstance(content, dict):
        if "text" in content:
            return extract_text(content["text"])
        elif "content" in content:
            return extract_text(content["content"])
    return str(content) if content is not None else ""

def chat_with_hf(messages, hf_token=None, model_name='Qwen/Qwen2.5-72B-Instruct', temperature=0.7, resume_text=None, target_role=None, required_skills=None):
    """
    Sends a conversation history list to Hugging Face and returns the response.
    messages format: [{"role": "user"|"assistant", "content": "string"}]
    """
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token or hf_token.strip() in ("", "YOUR_HF_TOKEN"):
        return "Error: Hugging Face API token is missing. Please configure HF_TOKEN in your .env or Settings."

    try:
        # Search relevant notes if available (local templates / action verbs)
        raw_last_query = messages[-1]['content'] if messages else ""
        last_query = extract_text(raw_last_query)
        context = ""
        try:
            retrieved_docs = search_notes(last_query, top_k=2) if last_query else []
            context = "\n".join(retrieved_docs)
        except Exception as e:
            print(f"RAG search error (ignored): {e}")

        system_instruction = """You are NaanChalant AI, a premium conversational assistant.
You are helping the user optimize their resume for applicant tracking systems (ATS).
"""
        if resume_text:
            system_instruction += f"\nHere is the user's resume text:\n---\n{resume_text}\n---\n"
        if target_role:
            system_instruction += f"\nTarget Job Role: {target_role}\n"
        if required_skills:
            system_instruction += f"\nAdditional Required Skills / Job Description:\n---\n{required_skills}\n---\n"

        system_instruction += """
Provide advice, rewrite bullet points using the STAR method, suggest bullet points for missing skills, and guide the user step-by-step on optimizing their resume for this role. Keep your answers formatting-friendly and easy to copy.
"""
        if context:
            system_instruction += f"\nAdditional retrieved general templates / guide notes:\n{context}\n"

        client = InferenceClient(model=model_name, token=hf_token)
        
        # Convert standard role list to OpenAI-style structures
        formatted_messages = [{"role": "system", "content": system_instruction}]
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'assistant'
            msg_content = extract_text(msg['content'])
            formatted_messages.append({
                'role': role,
                'content': msg_content
            })

        response = client.chat.completions.create(
            messages=formatted_messages,
            max_tokens=1024,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error during Hugging Face generation: {e}"