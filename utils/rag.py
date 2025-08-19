import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Import the new NLP processor
from utils.nlp_processor import process_text_with_nlp

def get_text_chunks(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # New: Process each chunk with spaCy and add metadata
        for chunk in chunks:
            nlp_data = process_text_with_nlp(chunk.page_content)
            chunk.metadata.update({
                "entities": str(nlp_data["entities"]),
                "pos_tags": str(nlp_data["pos_tags"])
            })
            
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to load or split PDF: {str(e)}")

def get_vector_store(text_chunks, embeddings_model):
    try:
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings_model)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")
