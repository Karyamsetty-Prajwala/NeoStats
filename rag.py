import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def get_text_chunks(file_path):
    """Loads a PDF and splits it into text chunks."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to load or split PDF: {str(e)}")

def get_vector_store(text_chunks, embeddings_model):
    """Creates a FAISS vector store from text chunks and embeddings."""
    try:
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings_model)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")