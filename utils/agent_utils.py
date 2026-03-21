import streamlit as st

# Safe import for AgentExecutor (handles diff langchain versions)
try:
    from langchain.agents import AgentExecutor
except ImportError:
    try:
        from langchain.agents.agent_executor import AgentExecutor
    except ImportError:
        # Emergency dummy fallback if langchain is completely mangled
        class AgentExecutor:
            def __init__(self, *args, **kwargs): pass
            def invoke(self, *args, **kwargs): return {"output": "Agent initialization failed. Please check dependencies."}

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from utils.search_utils import search_web
from utils.rag_utils import retrieve_context
from utils.analysis_utils import analyze_startup_csv
import langchain
import langchain_core
print(f"DEBUG: LangChain version: {langchain.__version__}")
print(f"DEBUG: LangChain Core version: {langchain_core.__version__}")

from models.llm import get_llm

@tool
def search_the_web(query: str) -> str:
    """Searches the live internet for recent news, funding rounds, and latest facts about Indian startups."""
    return search_web(query)

@tool
def query_knowledge_base(query: str) -> str:
    """Searches the local PDF knowledge base for specific facts, policies, rules, and definitions about Indian startups."""
    return retrieve_context(query)

@tool
def analyze_startup_data(query: str) -> str:
    """Queries the local startup_funding CSV dataset to perform mathematical aggregations, count startups, find top investors, and get funding totals."""
    return analyze_startup_csv(query)

def get_agent_executor(provider: str = None, response_instructions: str = "") -> AgentExecutor:
    llm = get_llm(provider)
    tools = [search_the_web, query_knowledge_base, analyze_startup_data]
    
    system_msg = (
        "You are the Indian Startup Intelligence Copilot, an autonomous assistant. "
        "You MUST use your tools to answer user queries accurately. "
        "If the user asks about recent events, use search_the_web. "
        "If they ask about CSV data or funding records, use analyze_startup_data. "
        "If they ask about policies or documents, use query_knowledge_base. "
        "CRITICAL: Always format financial amounts in Indian Rupees (₹) by default. "
        "{uploaded_file_instructions}\n"
    )
    if response_instructions:
        system_msg += f"\nCRITICAL RESPONSE FORMAT: {response_instructions}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    try:
        from langchain.agents import create_tool_calling_agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    except Exception as e:
        print(f"Tool calling init failed: {e}")
        try:
            from langchain.agents import create_react_agent
            from langchain import hub
            react_prompt = hub.pull("hwchase17/react")
            react_prompt = react_prompt.partial(uploaded_file_instructions="{uploaded_file_instructions}")
            agent = create_react_agent(llm, tools, react_prompt)
            return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        except Exception as e2:
            print(f"React fallback failed: {e2}")
            # Final dummy fallback so it doesn't crash app
            return AgentExecutor()
