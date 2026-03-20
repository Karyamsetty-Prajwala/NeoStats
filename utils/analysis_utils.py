import pandas as pd
import plotly.express as px
import streamlit as st
import os
import config.config as config
from models.llm import generate_response

@st.cache_data
def load_data():
    """Load the Kaggle Indian Startup Funding CSV."""
    if not os.path.exists(config.DATA_PATH):
        st.warning(f"Dataset not found at {config.DATA_PATH}")
        return pd.DataFrame()
    
    try:
        # Load the CSV, handling common encoding issues
        df = pd.read_csv(config.DATA_PATH, encoding='utf-8')
        
        # Clean basic column names
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
        
        # Attempt to clean 'amount_in_usd' column if it exists
        if 'amount_in_usd' in df.columns:
            df['amount_in_usd'] = df['amount_in_usd'].replace(r'[^\d.]', '', regex=True)
            df['amount_in_usd'] = pd.to_numeric(df['amount_in_usd'], errors='coerce')
            
        # Standardize date if present
        if 'date' in df.columns or 'date_dd/mm/yyyy' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'date_dd/mm/yyyy'
            df['year'] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors='coerce').dt.year
            df['year'] = df['year'].fillna(pd.to_datetime(df[date_col], errors='coerce').dt.year)
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def analyze_dataset(query: str, df: pd.DataFrame) -> tuple:
    """
    Analyzes the dataframe and returns a markdown response and a plotly figure (if any).
    We use the LLM to interpret the question, figure out the data, and we do the actual aggregation here.
    """
    if df.empty:
        return "Dataset is empty or not loaded.", None

    schema = df.dtypes.to_dict()
    head = df.head(3).to_dict()
    
    # Prompt the LLM to figure out what operation to perform
    prompt = f"""
You are a Data Analyst for Indian Startups.
The user asked: "{query}"

We have a pandas dataframe with the following columns:
{list(df.columns)}

Schema: {schema}
Sample Data: {head}

Analyze the data based on the question and provide your findings. Mention any trends, anomalies, or root causes you know about the ecosystem that relates to this data. 

To help you generate a response, consider extracting the key intent. Since you don't have direct access to execute python, just provide the most educated analytical answer you can give based on the headers and sample data, combined with your general knowledge of the Indian Startup ecosystem.
"""
    response_text = generate_response(prompt)
    
    fig = None
    
    # Simple heuristic to generate Plotly charts if asked
    lower_query = query.lower()
    
    if "sector" in lower_query or "industry" in lower_query:
        if 'industry_vertical' in df.columns:
            top_sectors = df['industry_vertical'].value_counts().head(10).reset_index()
            top_sectors.columns = ['industry_vertical', 'count']
            fig = px.bar(top_sectors, x='industry_vertical', y='count', title='Top 10 Sectors by Deal Count')
            
    elif "city" in lower_query or "location" in lower_query:
        if 'city_location' in df.columns:
            top_cities = df['city_location'].value_counts().head(10).reset_index()
            top_cities.columns = ['city_location', 'count']
            fig = px.pie(top_cities, names='city_location', values='count', title='Funding Deals by City')
            
    elif "year" in lower_query or "trend" in lower_query:
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='count')
            fig = px.line(yearly, x='year', y='count', title='Funding Trend by Year')
            
    return response_text, fig
