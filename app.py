import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from .models.llm import get_chatgroq_model
from .models.embeddings import get_huggingface_embeddings
from .utils.rag import get_text_chunks, get_vector_store
from .utils.web_search import get_tavily_tool

def get_agent_response(agent_executor, messages):
    """Get response from the agent executor"""
    try:
        # Get the latest human message
        last_human_message = messages[-1]["content"] if messages else ""
        
        # Invoke the agent executor with the current user input
        response = agent_executor.invoke({"input": last_human_message})
        return response['output']
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

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
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    
    # Get configuration from environment variables or session state
    # Set up the response mode selection in the sidebar
    response_mode = st.sidebar.radio(
        "Select Response Mode:",
        ("Concise", "Detailed"),
        index=0
    )
    
    # Dynamic system prompt based on response mode
    base_prompt = "You are a helpful legal document assistant."
    if response_mode == "Concise":
        mode_prompt = "Provide short, summarized replies based on the context."
    else: # Detailed
        mode_prompt = "Provide expanded, in-depth responses with detailed explanations based on the context."
        
    system_prompt = f"{base_prompt} {mode_prompt} If the user asks about recent legal updates or information not in the documents, use the web search tool."
    
    # Initialize the LLM, embeddings, and web search tool once per session
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = get_chatgroq_model()
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = get_huggingface_embeddings()
    if "tavily_tool" not in st.session_state:
        st.session_state.tavily_tool = get_tavily_tool()
    
    st.sidebar.subheader("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF legal document", type=["pdf"])
    
    # Handle document upload and create RAG retriever
    if uploaded_file and "retriever" not in st.session_state:
        with st.spinner("Processing document..."):
            # Save the uploaded file to a temporary location
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the PDF and create a vector store
            text_chunks = get_text_chunks("temp.pdf")
            vector_store = get_vector_store(text_chunks, st.session_state.embeddings)
            st.session_state.retriever = vector_store.as_retriever()
            st.sidebar.success("Document processed and ready!")
    
    # Create the RAG tool and agent
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
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if st.session_state.chat_model:
        if prompt_input := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt_input)
            
            # Generate and display bot response
            with st.chat_message("assistant"):
                with st.spinner("Getting response..."):
                    response = get_agent_response(agent_executor, st.session_state.messages)
                    st.markdown(response)
            
            # Add bot response to chat history
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
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🔄 Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                # Remove temporary files and state
                if os.path.exists("temp.pdf"):
                    os.remove("temp.pdf")
                if "retriever" in st.session_state:
                    del st.session_state.retriever
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
