import os
import hashlib
import shutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(".env")

# Global chatbot instance (per uploaded file)
chatbot_instance = None

# ==================== RAG COMPONENT ====================

class PDFChatbot:
    def __init__(self, file_path: str):
        self.file_path = file_path

        # 🔑 Unique ID per file
        self.pdf_id = hashlib.md5(file_path.encode()).hexdigest()

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def load_and_process_pdf(self):
        print("Loading document...")

        ext = os.path.splitext(self.file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(self.file_path)
        elif ext == ".txt":
            loader = TextLoader(self.file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(self.file_path)
        else:
            raise ValueError("Unsupported file type")

        documents = loader.load()

        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        # 🔑 Attach metadata for filtering
        for chunk in chunks:
            chunk.metadata["pdf_id"] = self.pdf_id
            chunk.metadata["source_file"] = os.path.basename(self.file_path)

        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks):
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        print("Loading / updating shared vector store...")
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7,
                "filter": {"pdf_id": self.pdf_id}
            }
        )

        print("Vector store ready (metadata filtering enabled).")

    def retrieve_context(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        return "\n\n".join(
            f"[{d.metadata.get('source_file')}]\n{d.page_content.strip()}"
            for d in docs
        )

    def setup_rag_chain(self):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        template = """You are a helpful voice assistant that answers questions strictly using the provided document context.

Context:
{context}

Question:
{question}

Rules:
- Use ONLY the context above
- If the answer is not in the context, say:
  "I don't have enough information in the document to answer this question."
- Keep the answer concise and conversational

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            {
                "context": lambda q: self.retrieve_context(q),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def initialize(self):
        chunks = self.load_and_process_pdf()
        self.create_vectorstore(chunks)
        self.setup_rag_chain()
        print("✅ Chatbot initialized for this document.")

    def ask_question(self, question: str) -> str:
        return self.rag_chain.invoke(question)

# ==================== FASTAPI APP ====================

app = FastAPI(title="Document RAG Chatbot API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    query: str

class UploadResponse(BaseModel):
    message: str
    filename: str


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    global chatbot_instance

    allowed_extensions = (".pdf", ".txt", ".docx", ".doc")

    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, DOCX, and DOC files are allowed"
        )

    upload_dir = "./uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chatbot_instance = PDFChatbot(file_path)
        chatbot_instance.initialize()

        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=file.filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    global chatbot_instance

    if chatbot_instance is None:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Please upload a file first using /upload-pdf"
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer = chatbot_instance.ask_question(request.query)
        return QueryResponse(answer=answer, query=request.query)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Document RAG Chatbot API",
        "endpoints": {
            "/upload-pdf": "POST",
            "/query": "POST",
            "/health": "GET",
            "/docs": "Swagger UI"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot_instance is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
