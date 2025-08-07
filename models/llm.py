import os
from langchain_groq import ChatGroq

def get_chatgroq_model():
    
    try:
        # Get API key from environment variable
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        # Choose a valid Groq model name
        model_name = "llama3-70b-8192"  # ✅ One of the available Groq models

        groq_model = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
