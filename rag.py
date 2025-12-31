import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

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
            chunk_size=1000,  # Smaller chunks for better precision
            chunk_overlap=200,  # More overlap for context continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create vector store with embeddings"""
        print("Creating embeddings...")
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Latest OpenAI embedding model
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print("Building vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retriever with MMR for diversity
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diverse results
            search_kwargs={
                "k": 5,  # Retrieve more documents
                "fetch_k": 10,  # Fetch more candidates before MMR
                "lambda_mult": 0.7  # Balance between relevance and diversity
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
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template with better instructions
        template = """You are a helpful assistant that answers questions based on the provided context from a PDF document.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context contains relevant information, provide a detailed answer
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the document to answer this question"
- Quote specific parts of the context when relevant
- Be concise but thorough

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain using LCEL with proper context retrieval
        self.rag_chain = (
            {
                "context": lambda x: self.retrieve_context(x),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def initialize(self):
        """Initialize the chatbot"""
        chunks = self.load_and_process_pdf()
        self.create_vectorstore(chunks)
        self.setup_rag_chain()
        print("\nChatbot initialized! You can now ask questions about your PDF.")
    
    def ask_question(self, question, show_sources=False):
        """Ask a question about the PDF"""
        if not self.rag_chain:
            return "Please initialize the chatbot first!"
        
        # Show retrieved chunks if requested
        if show_sources:
            print("\n--- Retrieved Context ---")
            context = self.retrieve_context(question)
            print(context)
            print("--- End of Context ---\n")
        
        response = self.rag_chain.invoke(question)
        return response
    
    def test_retrieval(self, query):
        """Test what chunks are being retrieved for a query"""
        print(f"\nTesting retrieval for: '{query}'")
        print("="*60)
        docs = self.retriever.invoke(query)
        for i, doc in enumerate(docs, 1):
            print(f"\n[Chunk {i}] (Page {doc.metadata.get('page', 'Unknown')})")
            print(f"{doc.page_content[:200]}...")
            print("-"*60)


def main():
    print("=== PDF RAG Chatbot ===\n")
    
    # Get PDF path from user
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
        return
    
    # Initialize chatbot
    chatbot = PDFChatbot(pdf_path)
    chatbot.initialize()
    
    # Interactive chat loop
    print("\n" + "="*50)
    print("You can now ask questions about your PDF.")
    print("Commands:")
    print("  - Type your question to get an answer")
    print("  - Type 'sources' to show retrieved chunks with answers")
    print("  - Type 'test: <query>' to test chunk retrieval")
    print("  - Type 'exit' or 'quit' to end")
    print("="*50 + "\n")
    
    show_sources = False
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if question.lower() == 'sources':
            show_sources = not show_sources
            status = "enabled" if show_sources else "disabled"
            print(f"\nSource display {status}\n")
            continue
        
        if question.lower().startswith('test:'):
            test_query = question[5:].strip()
            chatbot.test_retrieval(test_query)
            continue
        
        if not question:
            continue
        
        answer = chatbot.ask_question(question, show_sources=show_sources)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()