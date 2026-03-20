import os
from supabase import create_client, Client
import streamlit as st
import config.config as config

# Cache the Supabase client to avoid recreating it
@st.cache_resource
def init_supabase() -> Client:
    """Initialize Supabase Client."""
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        st.warning("Supabase URL or Key is missing. Memory will fallback to local session state only.")
        return None
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

def save_message(user_id: str, session_id: str, role: str, content: str):
    """Save a chat message to Supabase."""
    sb = init_supabase()
    if sb:
        try:
            # Assumes a table named 'chat_messages' with columns: 
            # id, user_id, session_id, role, content, timestamp
            sb.table("chat_messages").insert({
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "content": content
            }).execute()
        except Exception as e:
            st.error(f"Error saving to Supabase: {e}")

def get_chat_history(user_id: str, session_id: str) -> list:
    """Fetch chat history for a specific session."""
    sb = init_supabase()
    if sb:
        try:
            response = sb.table("chat_messages") \
                .select("role", "content") \
                .eq("user_id", user_id) \
                .eq("session_id", session_id) \
                .order("created_at", desc=False) \
                .limit(config.MAX_CHAT_HISTORY) \
                .execute()
            
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error reading from Supabase: {e}")
            return []
    return []

def get_user_sessions(user_id: str) -> list:
    """Fetch all unique session IDs for a user to display in the sidebar."""
    sb = init_supabase()
    if sb:
        try:
            # A distinct select on session_id
            response = sb.table("chat_messages") \
                .select("session_id") \
                .eq("user_id", user_id) \
                .execute()
            
            # Extract unique sessions
            if response.data:
                sessions = list(set([row["session_id"] for row in response.data]))
                return sessions
            return []
        except Exception as e:
            st.error(f"Error getting sessions from Supabase: {e}")
            return []
    return []
