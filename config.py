import streamlit as st

def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

# ============================================================
# API KEYS — never commit these to GitHub
# ============================================================
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
SERPER_API_KEY = get_secret("SERPER_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")

# SUPABASE CONFIG
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]

# ============================================================
# LLM SETTINGS
# ============================================================
DEFAULT_LLM_PROVIDER = "openrouter"  # "openrouter", "gemini" or "groq"
GEMINI_MODEL = "gemini-1.5-flash-latest"
GROQ_MODEL = "llama-3.1-8b-instant"
OPENROUTER_MODEL = "arcee-ai/trinity-mini:free"

# ============================================================
# RAG SETTINGS
# ============================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_RETRIEVAL_DOCS = 4
EMBEDDING_MODEL = "models/embedding-001"

# ============================================================
# DATA PATHS
# ============================================================
DATA_PATH = "data/startup_funding.csv"
DOCS_PATH = "docs/"
VECTORSTORE_PATH = "vectorstore/"

# ============================================================
# RESPONSE MODES
# ============================================================
CONCISE_PROMPT = "Reply in maximum 3 bullet points. Be very brief."
DETAILED_PROMPT = "Give a thorough, detailed analysis with insights, trends, and recommendations."

# ============================================================
# APP SETTINGS
# ============================================================
APP_TITLE = "Indian Startup Intelligence Copilot"
APP_ICON = "🚀"
MAX_CHAT_HISTORY = 50
