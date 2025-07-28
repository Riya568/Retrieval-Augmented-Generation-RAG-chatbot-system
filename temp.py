import gradio as gr
import faiss
import numpy as np
import pickle
import logging
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import os

# Secure API Key Configuration
GEMINI_API_KEY = "AIzaSyAvTxnxC4RdBiv6qjQTYZOh-SgHfZZUBog"  
genai.configure(api_key=GEMINI_API_KEY)

# Configure Logging
logging.basicConfig(level=logging.INFO)

# FastAPI Initialization
app = FastAPI()

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # Embedding dimension of MiniLM
faiss_index_path = "faiss_index.bin"
texts_path = "texts.pkl"

# Global Variables
index = None
texts = []

# Request Model for POST requests
class QueryRequest(BaseModel):
    query: str

# Extract Text from PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Convert Text to Embeddings
def embed_text(text):
    sentences = text.split(". ")
    embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
    return sentences, embeddings

# Load PDFs and Store in FAISS (if not already stored)
def load_pdfs(pdf_paths):
    global index, texts
    if os.path.exists(faiss_index_path) and os.path.exists(texts_path):
        logging.info("Loading existing FAISS index and texts...")
        index = faiss.read_index(faiss_index_path)
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
        logging.info(f"Loaded FAISS index with {len(texts)} sentences.")
        return
    
    index = faiss.IndexFlatL2(dimension)
    texts = []
    
    processed_count = 0
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            logging.info(f"Processing {pdf_path}...")
            try:
                text = extract_text_from_pdf(pdf_path)

                if not text:
                    logging.warning(f"No text extracted from {pdf_path}. Skipping...")
                    continue

                sentences, embeddings = embed_text(text)

                if len(sentences) == 0:
                    logging.warning(f"No sentences found in {pdf_path}. Skipping...")
                    continue

                texts.extend(sentences)
                index.add(embeddings)
                processed_count += 1
                logging.info(f"Indexed {len(sentences)} sentences from {pdf_path}.")
            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
                continue
        else:
            logging.warning(f"PDF file not found: {pdf_path}")
    
    if processed_count == 0:
        logging.error("No PDFs were successfully processed!")
        return
    
    # Save FAISS index and texts for future use
    faiss.write_index(index, faiss_index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)
    logging.info(f"FAISS index and texts saved successfully. Total sentences: {len(texts)}")

# Retrieve Relevant Context
def retrieve_relevant_text(query, k=20):
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No PDF data loaded. Upload PDFs first.")

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)

    if len(I[0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    retrieved_texts = [texts[i] for i in I[0] if i < len(texts)]
    logging.info(f"Retrieved Context: {retrieved_texts}")

    return " ".join(retrieved_texts)

# Generate Response from Gemini
def generate_gemini_response(query, context):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        formatted_prompt = f"""
        You are an AI expert. Provide a **concise and accurate definition** of the given topic using the provided context.

        **Question:** {query}

        **Context:**  
        {context}

        **Instructions:**  
        - Begin with a **clear one-line definition**.  
        - Follow with 2-3 **bullet points** for better understanding.  
        - **Do not discuss terminology issues—focus on definition.**  

        Provide a structured answer.
        """

        response = model.generate_content(formatted_prompt)

        if not response.text:
            return "I couldn't find enough details."

        logging.info(f"Gemini Response: {response.text.strip()}")
        return response.text.strip()
    
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return "I couldn't process your request."

# API Routes
@app.get("/")
def home():
    return {"message": "Welcome to the AI Chatbot API"}

@app.post("/ask/")
def ask_question_post(request: QueryRequest):
    user_query = request.query.strip()
    relevant_text = retrieve_relevant_text(user_query)
    response = generate_gemini_response(user_query, relevant_text)
    return {"response": response}

# Load PDFs on Startup (Only once)
pdf_paths = [
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\6-sigma-handnbook.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Artificial Intelligence Quantum (distinylearn.com).pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Big Data Analytics.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Big Data Principles and Paradigms.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Big Data Quantum.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Cloud Computing 4th yr.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Cloud Computing Quantum (distinylearn.com).pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Cryptography and network security IT.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\CYBER SECURITY Aktu Quantum.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Data Warehousing Data & Mining Quantum (distinylearn.com).pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Deep learning4thyr.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\green computing 1.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\green computing.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Green_Computing.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Machine learning 4th yr.pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Natural Language Processing Quantum (distinylearn.com).pdf",
    r"C:\Users\rg817\OneDrive\Desktop\QA_System\books\Six-Sigma-A-Complete-Step-by-Step-Guide.pdf"
]

logging.info("Initializing FAISS index and loading PDFs (if needed)...")
load_pdfs(pdf_paths)
logging.info("Initialization Complete!")

# Gradio Interface
def chatbot(query):
    relevant_text = retrieve_relevant_text(query)
    response = generate_gemini_response(query, relevant_text)
    return response

demo = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="RAG Chatbot",
                    description="Ask me questions based on the uploaded PDFs!")

# Run FastAPI & Gradio in Parallel
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_fastapi).start()
demo.launch(share=True)
