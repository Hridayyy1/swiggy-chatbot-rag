# Swiggy Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from the Swiggy Annual Report.

This project processes a PDF document, converts the content into embeddings using Sentence Transformers, stores them in a ChromaDB vector database, and retrieves relevant information when a user asks a question. The retrieved context is then used to generate a meaningful answer.

The system is built with a FastAPI backend and a simple chatbot user interface.

---

# Project Overview

Large documents such as annual reports contain a lot of information, but searching them manually is inefficient.

This project solves that problem using a **Retrieval-Augmented Generation (RAG) pipeline**.

The system performs the following steps:

1. Extract text from the PDF document
2. Split the text into smaller chunks
3. Convert each chunk into embeddings using a transformer model
4. Store the embeddings in a ChromaDB vector database
5. When a user asks a question, retrieve the most relevant chunks
6. Send the context to the LLM to generate an answer

This allows the chatbot to answer questions directly from the document.

---

# Tech Stack

Backend

* FastAPI
* Python

AI / NLP

* Sentence Transformers
* Embedding models

Vector Database

* ChromaDB

PDF Processing

* PyPDF2

Frontend

* HTML
* CSS
* JavaScript

---

# Project Structure

swiggy-chatbot-rag

server.py
FastAPI backend that handles PDF processing, embedding generation, vector search, and chatbot responses.

index.html
Frontend chatbot interface.

requirements.txt
List of Python dependencies required to run the project.

README.md
Project documentation.

.gitignore
Files that should not be pushed to GitHub.

---

# How the RAG Pipeline Works

1. The PDF document is loaded using PyPDF2
2. The text is split into smaller chunks
3. Each chunk is converted into a vector embedding using Sentence Transformers
4. Embeddings are stored in ChromaDB
5. When a user asks a question:

   * The question is converted into an embedding
   * ChromaDB retrieves the most relevant chunks
   * The retrieved context is passed to the LLM
6. The LLM generates a response based on the retrieved information

This architecture allows the chatbot to provide answers grounded in the document instead of hallucinating.

---

# Installation Guide

## 1. Clone the Repository

git clone https://github.com/Hridayyy1/swiggy-chatbot-rag.git

cd swiggy-chatbot-rag

---

## 2. Create a Virtual Environment (Recommended)

python -m venv venv

Activate the environment

Mac/Linux

source venv/bin/activate

Windows

venv\Scripts\activate

---

## 3. Install Dependencies

pip install -r requirements.txt

---

# Running the Application

Start the FastAPI server

uvicorn server --reload

The API server will start at

http://127.0.0.1:8000

FastAPI automatically provides API documentation at

http://127.0.0.1:8000/docs

---

# Running the Chatbot Interface

Open the file

index.html

in your browser.

The UI will connect to the FastAPI backend and allow you to ask questions about the document.

---

# Example Questions

You can try questions such as:

* What is Swiggy's revenue?
* What business segments does Swiggy operate in?
* What are the growth strategies mentioned in the report?
* What challenges does Swiggy discuss?

---

# Key Features

Document Question Answering
Allows users to ask questions directly from a large PDF document.

Semantic Search
Uses embeddings to find the most relevant document sections.

Vector Database
Efficient similarity search using ChromaDB.

FastAPI Backend
High-performance API for handling requests.

Interactive Chatbot UI
Simple and clean interface for asking questions.

---

# Future Improvements

Possible improvements include:

* Streaming responses
* Better UI with React
* Multi-document support
* Deployment using Docker
* Hosting on cloud platforms
* Authentication for users

---

# Author

Hriday Ranawat

B.Tech Information Technology
VJTI Mumbai

GitHub
https://github.com/Hridayyy1

LinkedIn
https://www.linkedin.com/in/hriday-ranawat-b71b50328/

---

# License

This project is open source and available under the MIT License.
