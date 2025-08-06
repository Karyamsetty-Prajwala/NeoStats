import os
from langchain_community.tools.tavily_research import TavilySearchResults



def get_tavily_tool():
    """Initializes and returns the Tavily web search tool."""
    try:
        # Get the API key from environment variables
        api_key = os.getenv("WEB_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("WEB_SEARCH_API_KEY environment variable not set.")
        
        tavily_tool = TavilySearchResults(api_key=api_key)
        return tavily_tool
    except Exception as e:
        # [cite_start]Wrap all functional code in try/except blocks to catch and log errors. [cite: 54]
        raise RuntimeError(f"Failed to initialize Tavily Search Tool: {str(e)}")

if __name__ == '__main__':
    # Example usage
    try:
        tavily_tool = get_tavily_tool()
        print("Tavily Search Tool initialized successfully.")
    except RuntimeError as e:
        print(e)
