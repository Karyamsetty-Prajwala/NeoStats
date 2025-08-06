import streamlit as st
import os
import sys

# Core LangChain Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

# Imports for RAG functionality
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# LLM Model from Groq
from langchain_groq import ChatGroq

# ==============================================================================
# Consolidated Function Definitions
# ==============================================================================

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        # Initialize the Groq chat model with the API key
        groq_model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")

def get_huggingface_embeddings():
    """Initializes and returns a HuggingFace embedding model."""
    try:
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
        raise RuntimeError(f"Failed to initialize HuggingFace embeddings: {str(e)}")

def get_text_chunks(file_path):
    """Loads a PDF and splits it into text chunks."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to load or split PDF: {str(e)}")

def get_vector_store(text_chunks, embeddings_model):
    """Creates a FAISS vector store from text chunks and embeddings."""
    try:
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings_model)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

def get_tavily_tool():
    """Initializes and returns the Tavily web search tool."""
    try:
        api_key = os.getenv("WEB_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("WEB_SEARCH_API_KEY environment variable not set.")
        
        tavily_tool = TavilySearchResults(api_key=api_key)
        return tavily_tool
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Tavily Search Tool: {str(e)}")

def get_agent_response(agent_executor, messages):
    """Get response from the agent executor"""
    try:
        last_human_message = messages[-1]["content"] if messages else ""
        response = agent_executor.invoke({"input": last_human_message})
        return response['output']
    except Exception as e:
        return f"Error getting response: {str(e)}"

# ==============================================================================
# Streamlit App Logic
# ==============================================================================

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 📥 Installation
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 🛠️ Available Models
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    
    ## How to Use
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Troubleshooting
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    
    ---
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    
    response_mode = st.sidebar.radio(
        "Select Response Mode:",
        ("Concise", "Detailed"),
        index=0
    )
    
    base_prompt = "You are a helpful legal document assistant."
    if response_mode == "Concise":
        mode_prompt = "Provide short, summarized replies based on the context."
    else:
        mode_prompt = "Provide expanded, in-depth responses with detailed explanations based on the context."
        
    system_prompt = f"{base_prompt} {mode_prompt} If the user asks about recent legal updates or information not in the documents, use the web search tool."
    
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = get_chatgroq_model()
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = get_huggingface_embeddings()
    if "tavily_tool" not in st.session_state:
        st.session_state.tavily_tool = get_tavily_tool()
    
    st.sidebar.subheader("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF legal document", type=["pdf"])
    
    if uploaded_file and "retriever" not in st.session_state:
        with st.spinner("Processing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            text_chunks = get_text_chunks("temp.pdf")
            vector_store = get_vector_store(text_chunks, st.session_state.embeddings)
            st.session_state.retriever = vector_store.as_retriever()
            st.sidebar.success("Document processed and ready!")
    
    tools = [st.session_state.tavily_tool]
    if "retriever" in st.session_state:
        from langchain.tools.retriever import create_retriever_tool
        retriever_tool = create_retriever_tool(
            st.session_state.retriever,
            "legal_document_search",
            "Searches and returns documents regarding legal questions.",
        )
        tools.append(retriever_tool)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    agent = create_tool_calling_agent(st.session_state.chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.chat_model:
        if prompt_input := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)
            with st.chat_message("assistant"):
                with st.spinner("Getting response..."):
                    response = get_agent_response(agent_executor, st.session_state.messages)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("No API keys found in environment variables. Please check the Instructions page to set up your API keys.")

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        if page == "Chat":
            st.divider()
            if st.button("🔄 Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if os.path.exists("temp.pdf"):
                    os.remove("temp.pdf")
                if "retriever" in st.session_state:
                    del st.session_state.retriever
                st.rerun()
    
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
