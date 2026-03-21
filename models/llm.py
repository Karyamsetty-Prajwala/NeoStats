import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import config.config as config

def get_llm(provider: str = None):
    """
    Returns the LangChain chat model based on the selected provider.
    Now optimized to use OpenRouter for Gemini to avoid 404 errors.
    """
    if provider is None:
        provider = config.DEFAULT_LLM_PROVIDER

    provider = provider.lower()

    # UNIVERSAL GEMINI FIX: If user selects Gemini, we use OpenRouter's version
    # because the native SDK is hitting 404 errors on some accounts.
    if provider == "gemini" or provider == "openrouter":
        return ChatOpenAI(
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name=config.OPENROUTER_MODEL,
            temperature=0.7,
            default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Indian Startup Copilot"}
        )

    elif provider == "groq":
        return ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=0.7
        )
    
    else:
        # Emergency fallback for anything else
        return get_llm("openrouter")

def generate_response(prompt: str, provider: str = None) -> str:
    """Helper function for quick generations."""
    llm = get_llm(provider)
    response = llm.invoke(prompt)
    return response.content