import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY")