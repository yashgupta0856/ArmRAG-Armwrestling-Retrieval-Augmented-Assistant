from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import ollama
from rag_utils import rag, build_vector_db

# ------------------------------
# Load PDF + Build Vector DB
# ------------------------------
PDF_PATH = "app/pdfContent/document.pdf"
build_vector_db(PDF_PATH)

# ------------------------------
# FastAPI Setup
# ------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# HOME ROUTE
# ------------------------------
@app.get("/")
async def home():
    return {"message": "ArmRAG backend running locally using Ollama"}


# ------------------------------
# MAIN RAG ENDPOINT
# ------------------------------
@app.post("/ask")
async def ask_question(question: str = Form(...)):

    context, sources = rag(question)

    prompt = f"""
You are a STRICT RAG assistant.
Use ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{question}

RULES:
1. Do NOT guess.
2. If the term is missing, respond:
   "The term '{question}' is not explicitly mentioned in the document."
"""

    response = ollama.generate(
        model="mistral",   # OR tinyllama
        prompt=prompt
    )

    answer = response["response"]

    return {
        "answer": answer,
        "sources": sources
    }
