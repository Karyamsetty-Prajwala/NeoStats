import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional: OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# API Key for a Web Search Tool (e.g., Tavily, SerpAPI)
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY")