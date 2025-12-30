import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import shutil
import uvicorn

load_dotenv(".env")

# Global variable to store the chatbot instance
chatbot_instance = None

# ==================== RAG Components ====================

class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.rag_chain = None
        self.retriever = None
        
    def load_and_process_pdf(self):
        """Load PDF and split into chunks"""
        print("Loading PDF...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create vector store with embeddings"""
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print("Building vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )
        print("Vector store created successfully!")
    
    def format_docs(self, docs):
        """Format retrieved documents with metadata"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            page_num = doc.metadata.get('page', 'Unknown')
            content = doc.page_content.strip()
            formatted.append(f"[Source {i} - Page {page_num}]\n{content}")
        return "\n\n".join(formatted)
    
    def retrieve_context(self, question):
        """Retrieve and format context for the question"""
        docs = self.retriever.invoke(question)
        return self.format_docs(docs)
    
    def setup_rag_chain(self):
        """Setup RAG chain with OpenAI LLM using LCEL"""
        llm_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        template = """You are a helpful voice assistant that answers questions based on the provided context from a PDF document.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context contains relevant information, provide a detailed answer
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the document to answer this question"
- Keep your response concise and conversational for voice interaction
- Avoid complex formatting, just speak naturally

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {
                "context": lambda x: self.retrieve_context(x),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm_model
            | StrOutputParser()
        )
    
    def initialize(self):
        """Initialize the chatbot"""
        chunks = self.load_and_process_pdf()
        self.create_vectorstore(chunks)
        self.setup_rag_chain()
        print("\nChatbot initialized!")
    
    def ask_question(self, question):
        """Ask a question about the PDF"""
        if not self.rag_chain:
            return "Please initialize the chatbot first!"
        
        response = self.rag_chain.invoke(question)
        return response


# ==================== FastAPI Application ====================

app = FastAPI(title="PDF RAG Chatbot API")

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
    """Upload a PDF file and initialize the RAG chatbot"""
    global chatbot_instance
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    upload_dir = "./uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chatbot_instance = PDFChatbot(file_path)
        chatbot_instance.initialize()
        
        return UploadResponse(
            message="PDF uploaded and processed successfully",
            filename=file.filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """Query the uploaded PDF document"""
    global chatbot_instance
    
    if chatbot_instance is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded. Please upload a PDF first using /upload-pdf endpoint"
        )
    
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        answer = chatbot_instance.ask_question(request.query)
        return QueryResponse(answer=answer, query=request.query)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "PDF RAG Chatbot API",
        "endpoints": {
            "/upload-pdf": "POST - Upload a PDF file",
            "/query": "POST - Query the uploaded PDF",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot_instance is not None
    }


if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    print("Upload PDFs via: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")