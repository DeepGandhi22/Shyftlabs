from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

genai.configure(api_key='AIzaSyDnAj8EiibHA3Pd1XG7xnqUPFO3duGY1CE')
gemini = genai.GenerativeModel('gemini-1.5-pro')

# Keep track of all documents for BM25
all_documents = []

def process_document(file=None, url=None):
    global all_documents
    try:
        if file:
            if file.filename.endswith(".pdf"):
                pdf_reader = PdfReader(file.file)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            else:
                raise ValueError("Unsupported file type")
        elif url:
            # Validate URL format
            if not url.startswith(("http://", "https://")):
                raise ValueError("Invalid URL format")
            
            # Fetch URL content
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad status codes
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
        else:
            raise ValueError("No input provided")

        # Split text into chunks
        splits = text_splitter.split_text(text)
        
        # Create Document objects
        documents = [Document(page_content=text, metadata={"source": file.filename if file else url}) 
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
        return vector_store.similarity_search(query, k=5)
    elif search_type == "keyword":
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 5
        return bm25_retriever.get_relevant_documents(query)
    else:
        return []

def generate_answer(question: str, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Answer the question based only on the following context:
    {context}
    
    Question: {question}"""
    
    try:
        response = gemini.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text.encode("utf-8")  # Ensure chunks are bytes
    except Exception as e:
        yield f"Error generating answer: {str(e)}".encode("utf-8")