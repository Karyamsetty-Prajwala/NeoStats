-- ====================================================
-- SUPABASE SCHEMA SETUP FOR NEOSTATS AI COPILOT
-- Run this in your Supabase SQL Editor (Project > SQL Editor)
-- ====================================================

-- 1. Table for chat messages (powers the ChatGPT-style sidebar)
CREATE TABLE IF NOT EXISTS chat_messages (
    id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     text NOT NULL,
    session_id  text NOT NULL,
    role        text NOT NULL CHECK (role IN ('user', 'assistant')),
    content     text NOT NULL,
    created_at  timestamptz DEFAULT now()
);

-- Index for fast per-user, per-session queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_session 
    ON chat_messages(user_id, session_id, created_at);

-- 2. Turn off Row Level Security for simplicity (suitable for demo/interview)
--    If you want stricter security, enable RLS and add policies.
ALTER TABLE chat_messages DISABLE ROW LEVEL SECURITY;
