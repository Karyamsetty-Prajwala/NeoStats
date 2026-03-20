import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embeddings
import config.config as config

def setup_vector_store():
    """
    Loads all PDFs in docs/, chunk them, and builds a FAISS vector store local index.
    If the index already exists, it just loads it.
    """
    os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
    index_name = "startup_docs"
    index_path = os.path.join(config.VECTORSTORE_PATH, index_name)

    embeddings = get_embeddings()

    # Check if vectorstore exists
    if os.path.exists(index_path):
        print(f"Loading existing FAISS vector store from {index_path}")
        vectorstore = FAISS.load_local(
            config.VECTORSTORE_PATH, 
            embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True # required for FAISS local load in trusted env
        )
        return vectorstore
    
    print(f"Creating new FAISS vector store from {config.DOCS_PATH}")
    loader = PyPDFDirectoryLoader(config.DOCS_PATH)
    docs = loader.load()

    if not docs:
        print(f"Warning: No PDF documents found in {config.DOCS_PATH}. RAG will have empty context.")
        # Create an empty vectorstore with a dummy document so it doesn't crash
        from langchain_core.documents import Document
        docs = [Document(page_content="No documents available in knowledge base.", metadata={"source": "system"})]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(config.VECTORSTORE_PATH, index_name)
    
    return vectorstore

def get_retriever():
    """Returns a retriever interface for the local vector store."""
    vectorstore = setup_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": config.MAX_RETRIEVAL_DOCS})

def retrieve_context(query: str) -> str:
    """Helper method to format retrieved docs into a context string."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant documents found."
        
    context = ""
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "?")
        context += f"\n--- Document {idx} (Source: {source}, Page: {page}) ---\n"
        context += doc.page_content + "\n"
        
    return context
