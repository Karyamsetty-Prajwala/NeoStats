import os
from dotenv import load_dotenv
load_dotenv()

# Get API key from .env
api_key = os.getenv("TAVILY_API_KEY")

# Import TavilySearch AFTER loading the API key
from langchain_tavily import TavilySearch

# Initialize Tavily Tool
tavily_tool = TavilySearch(api_key=api_key)

# Other necessary imports
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from models.llm import get_chatgroq_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool

# ==============================================================================
# Function Definitions
# ==============================================================================

def get_openai_embeddings():
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI embeddings: {str(e)}")

def get_text_chunks(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to load or split PDF: {str(e)}")

def get_vector_store(text_chunks, embeddings_model):
    try:
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings_model)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

def get_tavily_tool():
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in environment variables.")
        tavily_tool = TavilySearch(api_key=api_key)
        return tavily_tool
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Tavily Search Tool: {str(e)}")

def get_agent_response(agent_executor, messages):
    try:
        last_human_message = messages[-1]["content"] if messages else ""
        response = agent_executor.invoke({"input": last_human_message})
        return response['output']
    except Exception as e:
        return f"Error getting response: {str(e)}"

# ==============================================================================
# Streamlit App Pages
# ==============================================================================

def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("""
    ## 📅 Installation
    ```bash
    pip install -r requirements.txt
    ```

    ## API Key Setup
    - [OpenAI](https://platform.openai.com/api-keys)
    - [Groq](https://console.groq.com/keys)
    - [Google Gemini](https://aistudio.google.com/app/apikey)

    ## How to Use
    1. Upload a legal document on the **Chat** page.
    2. Ask questions about it.

    ---
    Navigate to the **Chat** page to start.
    """)

def chat_page():
    st.title("🤖 AI ChatBot")

    response_mode = st.sidebar.radio("Select Response Mode:", ("Concise", "Detailed"), index=0)

    base_prompt = "You are a helpful legal document assistant."
    mode_prompt = "Provide short, summarized replies based on the context." if response_mode == "Concise" \
        else "Provide expanded, in-depth responses with detailed explanations based on the context."

    system_prompt = f"{base_prompt} {mode_prompt} If the user asks about recent legal updates or information not in the documents, use the web search tool."

    if "chat_model" not in st.session_state:
        st.session_state.chat_model = get_chatgroq_model()

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = get_openai_embeddings()

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
        retriever_tool = create_retriever_tool(
            st.session_state.retriever,
            "legal_document_search",
            "Searches and returns documents regarding legal questions."
        )
        tools.append(retriever_tool)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

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
        st.info("No API keys found. Please check the Instructions page.")

# ==============================================================================
# Main App Entrypoint
# ==============================================================================

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

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
    else:
        chat_page()

if __name__ == "__main__":
    main()