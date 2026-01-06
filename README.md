📄 Voice-Enabled Document RAG Assistant (LiveKit + FastAPI)
🚀 Project Overview

This project is a real-time voice-enabled AI assistant that allows users to upload documents (PDF, TXT, DOC, DOCX) and ask spoken or text-based questions, receiving answers strictly derived from the uploaded document using Retrieval-Augmented Generation (RAG).

The system is composed of:

A LiveKit voice agent (agent.py) for real-time speech interaction

A FastAPI backend (app.py) that handles document ingestion, embedding, retrieval, and querying

The assistant follows strict rules to ensure that document-based answers are never hallucinated or rephrased.

🧠 Key Features

🎙️ Real-time Voice Assistant powered by LiveKit

📄 Supports PDF, TXT, DOC, and DOCX files

🔍 RAG-based document querying using LangChain

🧠 Uses OpenAI embeddings & LLMs

🧾 Exact-answer enforcement (no rephrasing or summarization)

🔊 Speech-to-Text (STT) & Text-to-Speech (TTS)

🌍 Multilingual turn detection

🔁 Interruption-safe conversations

📦 Persistent vector storage using ChromaDB

🏗️ System Architecture
User (Voice/Text)
      ↓
LiveKit Voice Agent (agent.py)
      ↓
FastAPI RAG Backend (app.py)
      ↓
LangChain + ChromaDB
      ↓
OpenAI Embeddings & LLM

📁 Project Structure
.
├── agent.py               # LiveKit voice agent
├── app.py                 # FastAPI RAG backend
├── uploaded_pdfs/         # Uploaded documents
├── chroma_db/             # Persistent vector database
├── .env                   # Environment variables
├── README.md              # Project documentation

⚙️ Environment Variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key

🧩 agent.py — Voice Agent (LiveKit)
Purpose

Handles:

Voice interaction

Tool calling

Document awareness

Speech synthesis and recognition

Key Components
🔧 Tools (Function Calling)
Tool Name	Description
check_document_status	Checks whether a document is uploaded
upload_pdf	Uploads a PDF file to the RAG backend
query_pdf	Queries the uploaded document
🤖 Assistant Rules

General Mode

Answers normal questions conversationally

Friendly and professional behavior

Document Mode (CRITICAL)

Always checks if a document exists

Always calls query_pdf for document questions

Never rephrases or expands answers

Speaks the response exactly as returned

🎙️ Voice Configuration
Component	Provider
STT	Deepgram Nova-2
LLM	OpenAI GPT-4.1-Mini
TTS	ElevenLabs Turbo v2.5
VAD	Silero
Turn Detection	Multilingual Model
Interruptions	Enabled
👋 Initial Greeting

The agent starts with a friendly welcome message and explains that it can:

Answer general questions

Work with documents (PDF, TXT, DOC, DOCX)

🧩 app.py — FastAPI RAG Backend
Purpose

Handles:

File uploads

Document processing

Embedding creation

Vector storage

Query answering

📄 Supported File Types

.pdf

.txt

.doc

.docx

🧠 RAG Pipeline (PDFChatbot)
1️⃣ Document Loading

Uses appropriate loaders:

PyPDFLoader

TextLoader

Docx2txtLoader

2️⃣ Chunking Strategy
Chunk Size: 2000 characters
Overlap: 500 characters
Separators: paragraphs, lines, sentences


Each chunk includes metadata:

pdf_id (unique per file)

source_file

3️⃣ Embeddings

Model: text-embedding-3-small

Provider: OpenAI

4️⃣ Vector Store

Database: ChromaDB

Persistence: ./chroma_db

Search Strategy: MMR (Maximal Marginal Relevance)

k = 5
fetch_k = 10
lambda_mult = 0.7


Metadata filtering ensures only the active document is queried.

5️⃣ RAG Prompt Rules

Uses ONLY provided context

If answer is missing:

I don't have enough information in the document to answer this question.


Concise and conversational output

🌐 API Endpoints
📤 Upload Document

POST /upload-pdf

Accepts PDF, TXT, DOC, DOCX

Initializes RAG pipeline

Response

{
  "message": "File uploaded and processed successfully",
  "filename": "example.pdf"
}

❓ Query Document

POST /query

Request

{
  "query": "What is this document about?"
}


Response

{
  "query": "What is this document about?",
  "answer": "Exact answer from the document"
}

❤️ Health Check

GET /health

{
  "status": "healthy",
  "chatbot_initialized": true
}

📘 API Docs

Swagger UI: /docs

▶️ How to Run the Project
1️⃣ Start FastAPI Backend
uv run app.py


Runs on:

http://localhost:8000

2️⃣ Start LiveKit Voice Agent
uv run agent.py console

🔒 Design Principles

❌ No hallucinations

❌ No rephrasing document answers

✅ Exact retrieval-based responses

✅ Metadata-isolated document querying

✅ Interruption-safe voice interactions

🧪 Use Cases

Voice-based document Q&A

Contract or policy explanation

Academic document querying

Hands-free document analysis

AI-powered customer support assistants

📌 Technologies Used

Python

LiveKit Agents

FastAPI

LangChain

ChromaDB

OpenAI (LLM + Embeddings)

Deepgram

ElevenLabs

Silero VAD

👤 Author

Atiq Ur Rehman
AI / Voice Agent / RAG Systems Developer