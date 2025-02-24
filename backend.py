from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import os
import requests
from dotenv import load_dotenv
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder

load_dotenv()

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="arxiv_paper",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

genai.configure(api_key='gemini_api_key')
gemini = genai.GenerativeModel('gemini-1.5-pro')

# Keep track of all documents for BM25
all_documents = []

def process_document(content_bytes: bytes = None, url: str = None, filename: str = None):
    global all_documents
    try:
        if content_bytes:
            # Process PDF from bytes
            pdf_reader = PdfReader(BytesIO(content_bytes))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            source = filename
        elif url:
            # Validate URL format
            if not url.startswith(("http://", "https://")):
                raise ValueError("Invalid URL format")
            
            # Fetch URL content
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            source = url
        else:
            raise ValueError("No input provided")

        # Split text into chunks
        splits = text_splitter.split_text(text)
        
        # Create Document objects
        documents = [Document(page_content=text, metadata={"source": source}) 
                     for text in splits]
        
        # Add to vector store and BM25
        vector_store.add_documents(documents)
        all_documents.extend(documents)
        return len(splits)
    
    except Exception as e:
        raise RuntimeError(f"Failed to process input: {str(e)}")

def hybrid_retriever(query: str, search_type: str):
    if not all_documents:
        raise ValueError("No documents have been processed yet. Please upload documents first.")
    
    if search_type == "semantic":
        # Re-rank results using cross-encoder
        context_docs = vector_store.similarity_search(query, k=10)
        pairs = [[query, doc.page_content] for doc in context_docs]
        scores = cross_encoder.predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, context_docs), reverse=True)]
        return ranked_docs[:5]
    elif search_type == "keyword":
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 5
        return bm25_retriever.get_relevant_documents(query)
    else:
        return []

def generate_answer(question: str, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are analyzing an academic paper. Answer with:
    1. Concise technical details from the context
    2. Never invent information
    3. Cite section numbers like [Section 3.2]
    
    Context: {context}
    
    Question: {question}"""
    
    try:
        response = gemini.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text.encode("utf-8")
    except Exception as e:
        yield f"Error generating answer: {str(e)}".encode("utf-8")