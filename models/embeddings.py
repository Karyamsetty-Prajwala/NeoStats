# models/embedding.py

import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Optional: Only disable SSL checks in development
if os.getenv("ENV") != "production":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

def get_gemini_embeddings():
   
    try:
        # Patch existing or create new event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply(loop)

        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_key
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini embeddings: {str(e)}")
