import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import config.config as config

def get_llm(provider: str = None):
    """
    Returns the LangChain chat model based on the selected provider.
    Defaults to the DEFAULT_LLM_PROVIDER in config.
    """
    if provider is None:
        provider = config.DEFAULT_LLM_PROVIDER

    provider = provider.lower()

    if provider == "openrouter":
        if not config.OPENROUTER_API_KEY:
            # Fallback to gemini if OpenRouter key is missing
            return get_llm("gemini")
            
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

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=0.7
        )
    
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")

def generate_response(prompt: str, provider: str = None) -> str:
    """Helper function for quick generations."""
    llm = get_llm(provider)
    response = llm.invoke(prompt)
    return response.content