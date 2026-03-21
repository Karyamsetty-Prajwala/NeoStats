import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from models.llm import get_llm
from utils.search_utils import search_web
from utils.rag_utils import retrieve_context
from utils.analysis_utils import analyze_startup_csv

class ManualAgentExecutor:
    """A custom, resilient agent loop that doesn't rely on langchain.agents."""
    def __init__(self, provider=None, response_instructions=""):
        self.provider = provider
        self.response_instructions = response_instructions
        self.tools_map = {
            "search_the_web": search_web,
            "query_knowledge_base": retrieve_context,
            "analyze_startup_data": analyze_startup_csv
        }
    
    def invoke(self, inputs):
        query = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        uploaded_file_instructions = inputs.get("uploaded_file_instructions", "")
        # Identity persistence: Get name from email if available
        user_display = st.session_state.get('user_email', 'User').split('@')[0]
        
        llm = get_llm(self.provider)
        
        # Define tools for the LLM
        tools_definitions = [
            {
                "name": "search_the_web",
                "description": "Searches the live internet for recent news, funding rounds, and latest facts about Indian startups.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            },
            {
                "name": "query_knowledge_base",
                "description": "Searches the local PDF knowledge base for specific facts, policies, rules, and definitions about Indian startups.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_startup_data",
                "description": "Queries the local startup_funding CSV dataset to perform mathematical aggregations, count startups, find top investors, and get funding totals.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools_definitions)
        
        # Prepare messages
        system_msg = (
            "You are the Indian Startup Intelligence Copilot, a friendly and professional autonomous assistant. "
            f"You are talking to {user_display}. Always greet them warmly and remember any personal details they share (like their name).\n\n"
            "You MUST use your tools to answer user queries accurately. "
            "If the user asks about recent events, use search_the_web. "
            "If they ask about CSV data or funding records, use analyze_startup_data. "
            "If they ask about policies or documents, use query_knowledge_base. "
            "CRITICAL: Always format financial amounts in Indian Rupees (₹) by default.\n"
            f"{uploaded_file_instructions}\n"
        )
        if self.response_instructions:
            system_msg += f"\nCRITICAL RESPONSE FORMAT: {self.response_instructions}"
            
        messages = [{"role": "system", "content": system_msg}]
        
        # Add history
        for m in chat_history:
            if isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                messages.append({"role": "assistant", "content": m.content})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Execution Loop (Max 5 turns)
        for _ in range(5):
            # Convert to langchain messages format for the call
            lc_msgs = []
            for m in messages:
                if m["role"] == "system": 
                    # Many models handle System best as a clear separate instruction
                    lc_msgs.append(HumanMessage(content=f"[SYSTEM INSTRUCTIONS]\n{m['content']}"))
                elif m["role"] == "user": 
                    lc_msgs.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant": 
                    lc_msgs.append(AIMessage(content=m.get("content") or "", tool_calls=m.get("tool_calls", [])))
                elif m["role"] == "tool": 
                    lc_msgs.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))
            
            try:
                resp = llm_with_tools.invoke(lc_msgs)
                
                # Check for tool calls
                if not resp.tool_calls:
                    return {"output": resp.content}
                
                # Execute tools
                messages.append({
                    "role": "assistant", 
                    "content": resp.content, 
                    "tool_calls": resp.tool_calls
                })
                
                for tool_call in resp.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    if tool_name in self.tools_map:
                        func = self.tools_map[tool_name]
                        result = func(tool_args.get("query", query))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": str(result)
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": f"Error: Tool {tool_name} not found."
                        })
            except Exception as e:
                return {"output": f"⚠️ Agent Loop Error: {e}"}
        
        return {"output": "Max iterations reached without final answer."}

def get_agent_executor(provider: str = None, response_instructions: str = ""):
    """Returns a manual agent executor instead of the broken langchain version."""
    return ManualAgentExecutor(provider=provider, response_instructions=response_instructions)
