from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config.config as config

def get_embeddings():
    """
    Returns the Google Generative AI Embeddings model for document chunking and vector search.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Cannot initialize embeddings.")

    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY
    )
