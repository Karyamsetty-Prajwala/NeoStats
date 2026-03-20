import streamlit as st
import uuid
import os
import pandas as pd
import io
from dotenv import load_dotenv

# Load .env file for local dev
load_dotenv()

# Load Streamlit secrets into env vars (for Streamlit Cloud)
try:
    for k, v in st.secrets.items():
        os.environ[k] = str(v)
except Exception:
    pass

import config.config as config

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
    [data-testid="stSidebar"] { background: #12122a; }
    .stTextInput > div > div > input {
        background: #1e1e3a; color: white;
        border: 1px solid #444; border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6c63ff, #48c9b0);
        color: white; border: none; border-radius: 8px;
        width: 100%; padding: 0.5rem; font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stChatMessage { border-radius: 12px; }
    [data-testid="stFileUploader"] {
        background: #1e1e3a; border: 1px dashed #6c63ff;
        border-radius: 10px; padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
defaults = {
    "user_id": None,
    "user_email": None,
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "response_mode": "Detailed",
    "llm_provider": "openrouter",
    "uploaded_file_context": None,   # stores extracted text from uploaded file
    "uploaded_file_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Lazy imports ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_dataframe():
    from utils.analysis_utils import load_data
    return load_data()

@st.cache_resource(show_spinner=False)
def get_supabase_cached():
    from utils.memory_utils import init_supabase
    return init_supabase()

def lazy_search(query):
    from utils.search_utils import search_web
    return search_web(query)

def lazy_rag(query):
    from utils.rag_utils import retrieve_context
    return retrieve_context(query)

def lazy_analyze(query, df):
    from utils.analysis_utils import analyze_dataset
    return analyze_dataset(query, df)

def lazy_llm(prompt, provider=None):
    from models.llm import generate_response
    return generate_response(prompt, provider)


# ─── FILE PROCESSING ──────────────────────────────────────────────────────────
def process_uploaded_file(uploaded_file) -> str:
    """Extract text content from uploaded CSV, PDF, DOCX, or TXT."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            summary = (
                f"Uploaded CSV: '{uploaded_file.name}'\n"
                f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                f"Columns: {list(df.columns)}\n\n"
                f"Sample (first 5 rows):\n{df.head(5).to_markdown()}\n\n"
                f"Basic Stats:\n{df.describe().to_markdown()}"
            )
            return summary
        elif name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded_file)
            parts = [f"Uploaded Excel: '{uploaded_file.name}' — Sheets: {xls.sheet_names}"]
            for sheet in xls.sheet_names[:3]:  # limit to 3 sheets
                df = xls.parse(sheet)
                parts.append(
                    f"\n--- Sheet: {sheet} ---\n"
                    f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Sample:\n{df.head(5).to_markdown()}\n"
                    f"Stats:\n{df.describe().to_markdown()}"
                )
            return "\n".join(parts)
        elif name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return f"Uploaded PDF: '{uploaded_file.name}'\n\n{text[:6000]}{'...[truncated]' if len(text) > 6000 else ''}"
        elif name.endswith((".docx",)):
            import docx as docx_lib
            doc = docx_lib.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return f"Uploaded Word Doc: '{uploaded_file.name}'\n\n{text[:6000]}{'...[truncated]' if len(text) > 6000 else ''}"
        elif name.endswith((".txt", ".md")):
            return f"Uploaded file: '{uploaded_file.name}'\n\n{uploaded_file.read().decode('utf-8', errors='ignore')[:6000]}"
        else:
            return f"Unsupported file type: {uploaded_file.name}."
    except Exception as e:
        return f"Error processing file '{uploaded_file.name}': {e}"


# ─── AUTH ─────────────────────────────────────────────────────────────────────
def login_form():
    st.markdown("""
    <div style='text-align:center; padding: 3rem 1rem 1rem;'>
        <h1 style='color:#6c63ff; font-size:2.8rem;'>🚀 Indian Startup Intelligence Copilot</h1>
        <p style='color:#aaa; font-size:1.1rem;'>Powered by AI · RAG · Live Web Search · Real Data</p>
    </div>
    """, unsafe_allow_html=True)

    sb = get_supabase_cached()
    _, col_m, _ = st.columns([1, 2, 1])
    with col_m:
        st.markdown("### Sign In to Continue")
        email    = st.text_input("📧 Email",    placeholder="you@example.com", key="auth_email")
        password = st.text_input("🔑 Password", type="password", placeholder="••••••••", key="auth_password")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", use_container_width=True):
                if not sb:
                    st.session_state.user_id    = "local-demo-user"
                    st.session_state.user_email = email or "demo@local"
                    st.rerun()
                else:
                    try:
                        res = sb.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user_id    = res.user.id
                        st.session_state.user_email = res.user.email
                        st.rerun()
                    except Exception as e:
                        err = str(e)
                        if "Email not confirmed" in err:
                            st.error("📬 Email not confirmed. Check your inbox or disable 'Confirm email' in Supabase Dashboard → Auth → Providers → Email.")
                        else:
                            st.error(f"Login failed: {err}")
        with c2:
            if st.button("Sign Up", use_container_width=True):
                if not sb:
                    st.warning("Supabase not connected – use Sign In for local demo.")
                else:
                    try:
                        sb.auth.sign_up({"email": email, "password": password})
                        st.success("✅ Signed up! Check your email, then Sign In.")
                    except Exception as e:
                        err = str(e)
                        st.warning(f"⏳ {err}") if "after" in err else st.error(f"Signup failed: {err}")
                        
        st.markdown(
            "<div style='text-align:center; padding: 1rem;'><p style='color:#aaa;'>— or —</p></div>", 
            unsafe_allow_html=True
        )
        if st.button("🚪 Continue as Guest (Bypass Login)", use_container_width=True):
            st.session_state.user_id    = "guest-user"
            st.session_state.user_email = email if email else "guest@neostats.ai"
            st.rerun()


def logout():
    sb = get_supabase_cached()
    if sb:
        try:
            sb.auth.sign_out()
        except Exception:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def build_sidebar():
    from utils.memory_utils import get_user_sessions, get_chat_history

    with st.sidebar:
        st.markdown(f"## {config.APP_ICON} AI Copilot")
        st.markdown(f"👤 **{st.session_state.user_email}**")
        st.divider()

        # Response mode
        st.session_state.response_mode = st.radio(
            "💡 Response Mode", ["Detailed", "Concise"], index=0
        )

        # LLM provider selector
        provider_map = {
            "🔵 OpenRouter (Free)": "openrouter",
            "🟣 Groq (Llama)":      "groq",
            "🔴 Gemini":            "gemini",
        }
        selected_label = st.selectbox(
            "🤖 LLM Provider",
            list(provider_map.keys()),
            index=0
        )
        st.session_state.llm_provider = provider_map[selected_label]

        st.divider()

        # File upload in sidebar
        st.subheader("📎 Upload a File")
        uploaded_files = st.file_uploader(
            "Upload CSV, Excel, PDF, Word (.docx), or TXT",
            type=["csv", "xlsx", "xls", "pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="sidebar_file_uploader"
        )
        if uploaded_files:
            names = [f.name for f in uploaded_files]
            names_str = ", ".join(names)
            if st.session_state.uploaded_file_name != names_str:
                with st.spinner(f"Processing {len(uploaded_files)} file(s)…"):
                    contexts = [process_uploaded_file(f) for f in uploaded_files]
                    st.session_state.uploaded_file_context = "\n\n---\n\n".join(contexts)
                    st.session_state.uploaded_file_name    = names_str
                st.success(f"✅ Loaded {len(uploaded_files)} file(s)")
        if st.session_state.uploaded_file_name:
            st.caption(f"📄 Active: `{st.session_state.uploaded_file_name}`")
            if st.button("🗑 Clear File", use_container_width=True):
                st.session_state.uploaded_file_context = None
                st.session_state.uploaded_file_name    = None
                st.rerun()

        st.divider()

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

        st.subheader("💬 Previous Chats")
        sessions = get_user_sessions(st.session_state.user_id)
        if sessions:
            for s_id in sessions[:10]:
                if st.button(f"🗂 {s_id[:8]}…", key=f"sess_{s_id}"):
                    st.session_state.session_id = s_id
                    history = get_chat_history(st.session_state.user_id, s_id)
                    st.session_state.messages = [
                        {"role": r["role"], "content": r["content"]} for r in history
                    ]
                    st.rerun()
        else:
            st.caption("_No previous chats yet._")

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()


# ─── QUERY ROUTER ─────────────────────────────────────────────────────────────
def handle_query(query: str):
    from utils.memory_utils import save_message

    lower_q    = query.lower()
    mode_instr = config.CONCISE_PROMPT if st.session_state.response_mode == "Concise" else config.DETAILED_PROMPT
    provider   = st.session_state.llm_provider
    df         = get_dataframe()
    context    = ""
    fig        = None

    # ── Uploaded file context (highest priority) ──────────────────────────────
    if st.session_state.uploaded_file_context:
        context += f"\n\n[UPLOADED FILE: {st.session_state.uploaded_file_name}]\n{st.session_state.uploaded_file_context}"

    # ── Live Web Search ───────────────────────────────────────────────────────
    if any(k in lower_q for k in ["news", "latest", "recently", "2024", "2025", "current"]):
        with st.status("🔍 Live web search…", expanded=False):
            try:
                context += f"\n\n[WEB SEARCH]\n{lazy_search(query)}"
            except Exception as e:
                context += f"\n\n[Web search error: {e}]"

    # ── RAG from docs/ ────────────────────────────────────────────────────────
    if any(k in lower_q for k in ["report", "pdf", "nasscom", "document", "case study", "neostats"]):
        with st.status("📄 Searching knowledge base…", expanded=False):
            try:
                context += f"\n\n[RAG KNOWLEDGE BASE]\n{lazy_rag(query)}"
            except Exception as e:
                context += f"\n\n[RAG error: {e}]"

    # ── CSV Data Analysis ─────────────────────────────────────────────────────
    if any(k in lower_q for k in ["funding", "sector", "startup", "year", "city", "investor",
                                    "top", "how many", "unicorn", "trend", "edtech", "fintech"]) \
            or (not context):
        with st.status("📊 Analysing startup data…", expanded=False):
            try:
                ans, fig_res = lazy_analyze(query, df)
                context += f"\n\n[CSV DATA ANALYSIS]\n{ans}"
                if fig_res:
                    fig = fig_res
            except Exception as e:
                context += f"\n\n[Data analysis error: {e}]"

    history_text = "".join(
        f"\n{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

    final_prompt = f"""You are the Indian Startup Intelligence Copilot.
CRITICAL RULE: When displaying quantitative values like funding amounts, ALWAYS format them in Indian Rupees (₹) by default (e.g., using Crores or Lakhs where appropriate) unless explicitly asked for dollars. Convert if necessary.

Conversation so far:{history_text}

User asked: "{query}"

Context from agents:
{context}

{mode_instr}
Use markdown. Bold key insights. Be professional and thorough.
"""

    with st.status("🤖 Generating response…", expanded=False):
        try:
            answer = lazy_llm(final_prompt, provider=provider)
        except Exception:
            try:
                answer = lazy_llm(final_prompt, provider="groq")
            except Exception as e:
                answer = f"⚠️ Error: {e}"

    return answer, fig


# ─── CHAT INTERFACE ───────────────────────────────────────────────────────────
def chat_interface():
    from utils.memory_utils import save_message

    st.markdown("### 💬 Chat with the Copilot")

    # Show active file banner if any
    if st.session_state.uploaded_file_name:
        st.info(f"📄 Chatting with file: **{st.session_state.uploaded_file_name}** — ask anything about it!", icon="📎")

    st.caption(f"Session: `{st.session_state.session_id[:16]}…` · LLM: `{st.session_state.llm_provider}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about Indian startups, or about your uploaded file…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        save_message(st.session_state.user_id, st.session_state.session_id, "user", prompt)

        response, fig = handle_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        save_message(st.session_state.user_id, st.session_state.session_id, "assistant", response)


# ─── ADMIN DASHBOARD ──────────────────────────────────────────────────────────
def admin_dashboard():
    st.markdown("### 📊 Admin Dashboard")
    sb = get_supabase_cached()
    if not sb:
        st.warning("Supabase not connected.")
        return
    try:
        resp = sb.table("chat_messages").select("id, role, created_at, user_id").execute()
        if not resp.data:
            st.info("No data yet. Start chatting to see analytics here.")
            return
        df = pd.DataFrame(resp.data)
        df["created_at"] = pd.to_datetime(df["created_at"])
        user_msgs = df[df["role"] == "user"]

        c1, c2, c3 = st.columns(3)
        c1.metric("📝 Total Queries",   len(user_msgs))
        c2.metric("👤 Unique Users",    df["user_id"].nunique())
        c3.metric("💬 Unique Sessions", df.groupby(["user_id"]).ngroups)

        st.divider()
        st.subheader("📈 Queries Over Time")
        daily = user_msgs.groupby(user_msgs["created_at"].dt.date).size().reset_index(name="count")
        st.bar_chart(daily.set_index("created_at"))

        st.subheader("📋 Recent Messages")
        st.dataframe(df.sort_values("created_at", ascending=False).head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.user_id:
        login_form()
    else:
        from streamlit_option_menu import option_menu
        build_sidebar()

        selected = option_menu(
            menu_title=None,
            options=["Chat", "Admin Dashboard"],
            icons=["chat-dots-fill", "graph-up-arrow"],
            orientation="horizontal",
            styles={
                "container":         {"background-color": "#1a1a2e"},
                "icon":              {"color": "#6c63ff", "font-size": "18px"},
                "nav-link-selected": {"background-color": "#6c63ff"},
            },
        )

        if selected == "Chat":
            chat_interface()
        elif selected == "Admin Dashboard":
            admin_dashboard()


if __name__ == "__main__":
    main()