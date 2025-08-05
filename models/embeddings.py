import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_huggingface_embeddings():
    """Initializes and returns a HuggingFace embedding model."""
    try:
        # We'll use a local, open-source model for simplicity
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    except Exception as e:
        # The project guidelines recommend wrapping functional code in try/except blocks 
        raise RuntimeError(f"Failed to initialize HuggingFace embeddings: {str(e)}")

if __name__ == '__main__':
    # Example of how to use the function
    try:
        embeddings_model = get_huggingface_embeddings()
        print("HuggingFace Embeddings model initialized successfully.")
    except RuntimeError as e:
        print(e)