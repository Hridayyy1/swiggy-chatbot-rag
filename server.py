from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

import os
import requests
import re
from dotenv import load_dotenv


# -------- Load environment variables --------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# -------- FastAPI setup --------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- GLOBAL VARIABLES --------
model = None
collection = None


# -------- TEXT PREPROCESSING --------
def preprocess_text(text):

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# -------- CLEAN LLM RESPONSE --------
def clean_response(text):

    text = text.replace("\\", "")
    text = text.replace("*", "")
    text = text.replace("_", "")

    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# -------- LOAD PDF --------
def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return preprocess_text(text)


# -------- SMART CHUNKING --------
def chunk_text(text, chunk_size=800, overlap=150):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -------- STARTUP EVENT (LOAD MODEL + VECTOR DB) --------
@app.on_event("startup")
async def startup_event():

    global model, collection

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Initializing ChromaDB...")
    chroma_client = chromadb.Client()

    collection = chroma_client.create_collection(name="swiggy_rag")

    if os.path.exists("swiggy_report.pdf"):

        print("Loading Swiggy Annual Report...")

        report_text = extract_text_from_pdf("swiggy_report.pdf")

        chunks = chunk_text(report_text)

        print(f"Total chunks created: {len(chunks)}")

        embeddings = model.encode(chunks).tolist()

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"id_{i}" for i in range(len(chunks))]
        )

        print("Vector database ready.")

    else:

        print("⚠ swiggy_report.pdf not found.")


# -------- REQUEST MODEL --------
class ChatRequest(BaseModel):
    message: str


# -------- CHAT ENDPOINT --------
@app.post("/chat")
async def chat(req: ChatRequest):

    question = req.message

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    context = "\n\n".join(results["documents"][0])

    system_prompt = f"""
You are an AI assistant that answers questions about the Swiggy Annual Report.

Instructions:
- Answer clearly and concisely.
- Use ONLY the information from the context.
- Use plain text only.
- Do not use markdown symbols (*, _, #).
- If financial numbers exist, extract them exactly.
- If the answer is not found, say:
"I could not find that information in the report."

Context:
{context}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()

    if "choices" not in result:

        return {
            "response": "LLM error",
            "debug": result
        }

    answer = result["choices"][0]["message"]["content"]

    answer = clean_response(answer)

    return {
        "response": answer,
        "sources": results["documents"][0]
    }


# -------- HOME PAGE --------
@app.get("/", response_class=HTMLResponse)
def home():

    with open("static/index.html") as f:
        return f.read()