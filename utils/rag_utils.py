import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    EnsembleRetriever = None
from langchain_community.retrievers import BM25Retriever
import pickle
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

    bm25_path = os.path.join(config.VECTORSTORE_PATH, "bm25_retriever.pkl")

    # Check if vectorstore exists
    if os.path.exists(index_path) and os.path.exists(bm25_path):
        print(f"Loading existing Hybrid vector store from {index_path}")
        vectorstore = FAISS.load_local(
            config.VECTORSTORE_PATH, 
            embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True # required for FAISS local load in trusted env
        )
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        return vectorstore, bm25_retriever
    
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
    
    # Store Sparse BM25
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = config.MAX_RETRIEVAL_DOCS
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    return vectorstore, bm25_retriever

def get_retriever():
    """Returns a hybrid ensemble retriever interface."""
    vectorstore, bm25_retriever = setup_vector_store()
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.MAX_RETRIEVAL_DOCS})
    
    if EnsembleRetriever:
        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
    else:
        # Fallback to standard vector retrieval if Hybrid is unavailable
        return faiss_retriever

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
