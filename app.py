import os
from dotenv import load_dotenv
load_dotenv()

# Import TavilySearch BEFORE it is used
from langchain_tavily import TavilySearch

# Get API key from .env AFTER importing TavilySearch
api_key = os.getenv("TAVILY_API_KEY")

# Other necessary imports
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader # Keep this import, as it's used
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from models.llm import get_chatgroq_model # Assuming models.llm exists and get_chatgroq_model is defined
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool


def get_openai_embeddings():
    """Initializes and returns an OpenAIEmbeddings model."""
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY")
        if not openai_key:
            # Display an error and stop execution if API key is missing
            st.error("OPENAI_API_KEY missing in Streamlit secrets. Please set it to use embeddings.")
            st.stop() # Stop the script execution
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
        st.stop() # Stop the script execution on embedding initialization failure


def get_text_chunks(file_path):
    """
    Loads a PDF and splits it into text chunks.
    Uses UnstructuredPDFLoader for better handling of various PDF types.
    Returns an empty list if no documents are loaded or an error occurs.
    """
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            st.warning("No text could be extracted from the uploaded PDF. It might be empty or scanned without OCR.")
            return [] # Return an empty list if no documents are loaded

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        st.error(f"Error loading or splitting PDF: {str(e)}")
        # import traceback # Uncomment for detailed debugging in console
        # print("Traceback for text chunking failure:\n", traceback.format_exc())
        return [] # Return an empty list on error


def get_vector_store(text_chunks, embeddings_model):
    """
    Creates and returns a FAISS vector store from text chunks.
    Handles cases where text_chunks might be empty.
    """
    try:
        if not text_chunks:
            st.warning("Cannot create vector store: No text chunks were provided.")
            return None # Return None if there are no chunks to process

        print(f"Number of chunks: {len(text_chunks)}")
        print(f"Embedding model: {embeddings_model}")
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings_model)
        return vector_store
    except Exception as e:
        import traceback
        print("Traceback for vector store failure:\n", traceback.format_exc())
        st.error(f"Failed to create vector store: {str(e)}. This might happen if the document is too small or unreadable.")
        return None # Return None on error


def get_tavily_tool():
    """Initializes and returns the Tavily Search Tool."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            st.error("TAVILY_API_KEY not set in environment variables. Please set it.")
            st.stop() # Stop the script execution
        tavily_tool = TavilySearch(api_key=api_key)
        return tavily_tool
    except Exception as e:
        st.error(f"Failed to initialize Tavily Search Tool: {str(e)}")
        st.stop() # Stop the script execution on tool initialization failure


def get_agent_response(agent_executor, messages):
    """Invokes the agent executor with the last human message."""
    try:
        # Extract chat history for the agent if needed, otherwise just the last message
        chat_history_for_agent = []
        for msg in messages:
            if msg["role"] == "user":
                chat_history_for_agent.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_for_agent.append(("ai", msg["content"]))

        # The agent executor typically expects 'input' and 'chat_history'
        response = agent_executor.invoke({
            "input": messages[-1]["content"], # Last human message
            "chat_history": chat_history_for_agent[:-1] # All messages except the very last user input
        })
        return response['output']
    except Exception as e:
        st.error(f"Error getting response from agent: {str(e)}")
        # import traceback # Uncomment for detailed debugging in console
        # print("Traceback for agent response failure:\n", traceback.format_exc())
        return "An error occurred while generating the response."


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


def chat_page():
    st.title("🤖 AI Legal Assistant")

    # Sidebar for response mode
    response_mode = st.sidebar.radio("Select Response Mode:", ("Concise", "Detailed"), index=0)

    # Dynamic system prompt based on response mode
    base_prompt = "You are a helpful legal document assistant."
    mode_prompt = "Provide short, summarized replies based on the context." if response_mode == "Concise" \
        else "Provide expanded, in-depth responses with detailed explanations based on the context."

    system_prompt = f"{base_prompt} {mode_prompt} If the user asks about recent legal updates or information not in the documents, use the web search tool."

    # Initialize models and tools only once
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = get_chatgroq_model() # Ensure this function handles its API key internally or via st.secrets

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = get_openai_embeddings() # This function now handles errors and stops execution

    if "tavily_tool" not in st.session_state:
        st.session_state.tavily_tool = get_tavily_tool() # This function now handles errors and stops execution

    st.sidebar.subheader("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF legal document", type=["pdf"])

    # Process uploaded file if it exists and retriever is not yet set
    if uploaded_file and "retriever" not in st.session_state:
        with st.spinner("Processing document..."):
            temp_file_path = "temp.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text_chunks = get_text_chunks(temp_file_path)
            
            # Only proceed if text chunks were successfully extracted
            if text_chunks:
                st.write(f"Number of text chunks: {len(text_chunks)}")
                vector_store = get_vector_store(text_chunks, st.session_state.embeddings)
                
                if vector_store: # Ensure vector_store was successfully created
                    st.session_state.retriever = vector_store.as_retriever()
                    st.sidebar.success("Document processed and ready!")
                else:
                    st.sidebar.error("Could not create document retriever. Please try another PDF.")
            else:
                st.sidebar.error("Failed to extract text from the document. Please ensure it's a readable PDF.")
            
            # Clean up the temporary file regardless of success or failure
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


    # Define tools for the agent
    tools = [st.session_state.tavily_tool]
    if "retriever" in st.session_state:
        retriever_tool = create_retriever_tool(
            st.session_state.retriever,
            "legal_document_search",
            "Searches and returns documents regarding legal questions."
        )
        tools.append(retriever_tool)

    # Construct the chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the agent and executor
    agent = create_tool_calling_agent(st.session_state.chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Added verbose for debugging

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if st.session_state.chat_model: # Only allow input if chat model is initialized
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
                st.rerun() # Corrected from st.experimental_rerun()

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
