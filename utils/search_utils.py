import requests
import json
import config.config as config

def search_web(query: str) -> str:
    """
    Perform a live web search using the Serper API.
    Used for answering queries about recent events not covered by the dataset or RAG docs.
    """
    if not config.SERPER_API_KEY:
        return "Error: Serper API key is missing. Cannot perform web search."
        
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "gl": "in",  # Google locale India
        "hl": "en"   # Language English
    })
    headers = {
        'X-API-KEY': config.SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse the JSON response to make it readable for the LLM
        results_str = f"Live Web Search Results for '{query}':\n\n"
        
        if "answerBox" in data and "snippet" in data["answerBox"]:
            results_str += f"**Answer Box:** {data['answerBox']['snippet']}\n\n"
            
        if "organic" in data:
            for item in data["organic"][:5]:  # Top 5 results
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No Snippet")
                link = item.get("link", "#")
                results_str += f"- **{title}**: {snippet} (Source: {link})\n"
                
        return results_str
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"
