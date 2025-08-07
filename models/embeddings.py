# embedding.py
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_gemini_embeddings():
    """Initializes and returns Google Gemini Embeddings with proper event loop handling."""
    try:
        # ✅ Manually create an event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_key
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini embeddings: {str(e)}")
