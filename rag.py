import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()


class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

        # 🔑 UNIQUE ID PER PDF (based on file path)
        self.pdf_id = hashlib.md5(pdf_path.encode()).hexdigest()

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def load_and_process_pdf(self):
        """Load PDF and split into chunks"""
        print("Loading PDF...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        # 🔑 ADD METADATA TO EACH CHUNK
        for chunk in chunks:
            chunk.metadata["pdf_id"] = self.pdf_id
            chunk.metadata["source_file"] = os.path.basename(self.pdf_path)

        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks):
        """Create / load vector store and apply metadata filtering"""
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

        # 🔹 ADD NEW DOCUMENTS (does NOT overwrite existing PDFs)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

        # 🔑 FILTER RETRIEVAL BY pdf_id
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7,
                "filter": {"pdf_id": self.pdf_id}  # ⭐ IMPORTANT
            }
        )

        print("Vector store ready (metadata filtering enabled).")

    def format_docs(self, docs):
        """Format retrieved documents"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source_file", "Unknown")
            formatted.append(
                f"[Source {i} | Page {page} | File: {source}]\n{doc.page_content.strip()}"
            )
        return "\n\n".join(formatted)

    def retrieve_context(self, question):
        docs = self.retriever.invoke(question)
        return self.format_docs(docs)

    def setup_rag_chain(self):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        template = """You are a helpful assistant that answers questions strictly based on the provided PDF context.

Context:
{context}

Question:
{question}

Rules:
- Use ONLY the given context
- If the answer is not in the context, say:
  "I don't have enough information in this document to answer this question."
- Quote the document when relevant

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
        print("\n✅ Chatbot initialized for this PDF only.")

    def ask_question(self, question, show_sources=False):
        if show_sources:
            print("\n--- Retrieved Context ---")
            print(self.retrieve_context(question))
            print("--- End Context ---\n")

        return self.rag_chain.invoke(question)

    def test_retrieval(self, query):
        print(f"\nTesting retrieval for: {query}")
        print("=" * 60)
        docs = self.retriever.invoke(query)
        for i, doc in enumerate(docs, 1):
            print(f"\n[Chunk {i}] File: {doc.metadata.get('source_file')}")
            print(doc.page_content[:200])
            print("-" * 60)


def main():
    print("=== PDF RAG Chatbot (Multi-PDF Safe) ===\n")

    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ PDF file not found!")
        return

    chatbot = PDFChatbot(pdf_path)
    chatbot.initialize()

    print("\nAsk questions about THIS PDF only.")
    print("Commands: sources | test: <query> | exit\n")

    show_sources = False

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if question.lower() == "sources":
            show_sources = not show_sources
            print(f"Source display {'ON' if show_sources else 'OFF'}")
            continue

        if question.startswith("test:"):
            chatbot.test_retrieval(question[5:].strip())
            continue

        if question:
            answer = chatbot.ask_question(question, show_sources)
            print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
