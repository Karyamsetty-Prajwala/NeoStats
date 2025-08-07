import os
from dotenv import load_dotenv
load_dotenv()
import pdfplumber
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import the patched get_gemini_embeddings function from your embeddings.py file
from model.embeddings import get_gemini_embeddings

# Import TavilySearch BEFORE it is used
from langchain_tavily import TavilySearch

# Get API key from .env AFTER importing TavilySearch
api_key = os.getenv("TAVILY_API_KEY")

# Other necessary imports
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from models.llm import get_chatgroq_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document

# ❌ REMOVE the old get_gemini_embeddings function from app.py
# The patched version from embeddings.py is now being used

def get_text_chunks_pdfplumber(file_path):
    with pdfplumber.open(file_path) as pdf:
        full_text = ''
        for page in pdf.pages:
            full_text += page.extract_text() + '\n'

    chunk_size = 2000
    overlap = 400
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    return chunks

def get_vector_store(text_chunks, embeddings_model):
    try:
        if not text_chunks:
            st.warning("Cannot create vector store: No text chunks were provided.")
            return None

        documents = [Document(page_content=chunk) for chunk in text_chunks]

        print(f"Number of chunks: {len(documents)}")
        print(f"Embedding model: {embeddings_model}")
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings_model)
        return vector_store
    except Exception as e:
        import traceback
        print("Traceback for vector store failure:\n", traceback.format_exc())
        st.error(f"Failed to create vector store: {str(e)}. This might happen if the document is too small or unreadable.")
        return None

def get_tavily_tool():
    """Initializes and returns the Tavily Search Tool."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            st.error("TAVILY_API_KEY not set in environment variables. Please set it.")
            st.stop()
        tavily_tool = TavilySearch(api_key=api_key)
        return tavily_tool
    except Exception as e:
        st.error(f"Failed to initialize Tavily Search Tool: {str(e)}")
        st.stop()

def get_agent_response(agent_executor, messages):
    """Invokes the agent executor with the last human message."""
    try:
        chat_history_for_agent = []
        for msg in messages:
            if msg["role"] == "user":
                chat_history_for_agent.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_for_agent.append(("ai", msg["content"]))

        response = agent_executor.invoke({
            "input": messages[-1]["content"],
            "chat_history": chat_history_for_agent[:-1]
        })
        return response['output']
    except Exception as e:
        st.error(f"Error getting response from agent: {str(e)}")
        return "An error occurred while generating the response."

system_prompt = """You are a helpful legal assistant. Your primary function is to answer questions based on the uploaded legal document.
You should also use the Tavily Search tool to find recent legal updates or supplementary information when the question requires it.
If the question is about a specific document, use the `legal_document_search` tool.
If the question requires general legal knowledge or web searches, use the `tavily_search_results_json` tool.
Always cite your sources and be concise and professional."""

def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("""
    ## 📅 Installation
    ```bash
    pip install -r requirements.txt
    ```
    ---
    ## API Key Setup
    - **OpenAI**: Get your key from [OpenAI API Keys](https://platform.openai.com/api-keys). Set it as `OPENAI_API_KEY` in Streamlit secrets.
    - **Groq**: Get your key from [Groq Console](https://console.groq.com/keys). Set it as `GROQ_API_KEY` in Streamlit secrets (or environment variables if `get_chatgroq_model` uses `os.getenv`).
    - **Tavily**: Get your key from [Tavily AI](https://tavily.com/). Set it as `TAVILY_API_KEY` in your `.env` file or Streamlit secrets.
    - **Google Gemini**: Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey). Set it as `GOOGLE_API_KEY` in Streamlit secrets.

    ---
    ## How to Use
    1. Upload a legal document on the **Chat** page.
    2. Ask questions about it.
    3. The bot can also search the web for recent legal updates if needed.

    ---
    Navigate to the **Chat** page to start.
    """)
def get_text_chunks(file_path):
    return get_text_chunks_pdfplumber(file_path)

def chat_page():
    st.title("🤖 AI Legal Assistant")

    st.sidebar.subheader("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF legal document", type=["pdf"])

    if "chat_model" not in st.session_state:
        try:
            st.session_state.chat_model = get_chatgroq_model()
        except Exception as e:
            st.error(str(e))
            return

    if "embeddings" not in st.session_state:
        try:
            # ✅ Using the patched function from embeddings.py
            st.session_state.embeddings = get_gemini_embeddings()
        except Exception as e:
            st.error(str(e))
            return

    if "tavily_tool" not in st.session_state:
        try:
            st.session_state.tavily_tool = get_tavily_tool()
        except Exception as e:
            st.error(str(e))
            return

    if uploaded_file and "retriever" not in st.session_state:
        with st.spinner("Processing document..."):
            temp_file_path = "temp.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            text_chunks = get_text_chunks(temp_file_path)
            if text_chunks:
                vector_store = get_vector_store(text_chunks, st.session_state.embeddings)
                if vector_store:
                    st.session_state.retriever = vector_store.as_retriever()
                    st.sidebar.success("Document processed!")
                else:
                    st.sidebar.error("Could not create vector store. Try another PDF.")
            else:
                st.sidebar.error("Text extraction failed. Upload a readable PDF.")

            os.remove(temp_file_path)

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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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
        st.info("No API keys found or model initialization failed. Please check the Instructions page and your API keys.")

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