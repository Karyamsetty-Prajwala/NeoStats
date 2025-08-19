**Lexibot-AI Legal Assistant**

**Features:**

*  PDF Upload – Upload legal documents and extract content.  
* Vector Embedding – Uses Google Gemini embeddings for semantic understanding.  
* Smart Search Tools – Integrates Tavily AI Search for real-time legal insights.  
* Response Modes – Choose between Concise or Detailed replies.  
* Contextual Risk Highlighter – Detects possible legal loopholes or areas of concern.  
* Fallback to Gemini Web Search – If the document lacks info, it retrieves data using Gemini.  
* Chat History & Rerun – Retains or resets chat state and document context.

# **Tech Stack**

**| Tool |                                    |Purpose |**

**|--------------|                           |-----------|**

| Streamlit |                            |UI and frontend |

| LangChain |                         | LLM orchestration |

| Google Generative AI |       |Embedding model |

| Tavily |                                  | Real-time web search |

| FAISS |                                  |Vector store |

| pdfplumber |                      |PDF parsing |

**SetUp Instructions:**

1. Clone the repository:  
   git clone https://github.com/your-username/neostats-legal-assistant.git  
   cd neostats-legal-assistant  
2. Install Dependencies  
   pip install \-r requirements.txt  
     
3. Create a .env File  
   GOOGLE\_API\_KEY=your\_google\_gemini\_key  
   TAVILY\_API\_KEY=your\_tavily\_key  
   GROQ\_API\_KEY=your\_groq\_api\_key  \# If you're using Groq LLM  
     
4. Run the App locally:  
    streamlit run [app.py](http://app.py)  
   	OR  
   Python \-m streamlit run [app.py](http://app.py)

**How It Works**

* Upload a legal document (PDF).

* It extracts text and creates chunks.

* These chunks are embedded using Gemini and stored in FAISS.

* Ask questions in the chat box.

* The agent responds using:

  * Document context (retriever tool),

  * Tavily AI for real-time search,

  * Google Gemini fallback if needed.

* Use the sidebar to toggle between Concise and Detailed answers.


**Folder Structure:**

├── app.py                  \# Main Streamlit app

├── models/

│      ├── embeddings.py       \# Gemini embedding wrapper

│      └── llm.py              \# Chat model loader

├── requirements.txt

├── .env                    \# API Keys (Not committed)

└── README.md

**Deployment:**  
To deploy on Streamlit Cloud:

1. Push your code to GitHub.

2. Go to Streamlit Cloud → New app.

3. Connect your repo and set environment variables in Secrets Manager.

   

   

   

