import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QA/QC LLAA Advisor", page_icon="🏗️", layout="wide")

# --- 1. SECURITY CONFIGURATION (GEMINI) ---
gemini_key = None
if "GEMINI_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_KEY"]
else:
    gemini_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

# --- 2. SEMANTIC MODEL LOADING (LIGHTWEIGHT ENGLISH MODEL) ---
@st.cache_resource
def load_llm_encoder():
    # Back to the fast, lightweight English-only model (~80MB)
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. ROBUST DATA LOADING ---
@st.cache_data
def load_data(file_path):
    try:
        # Reading with semicolon separator as per your generated CSV
        df = pd.read_csv(file_path, sep=';', engine='python', encoding='utf-8-sig')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    df.columns = df.columns.str.strip()
    
    # Map essential columns
    col_title = 'Title'
    col_desc = 'Description'
    col_cat = 'Knowledge Category'

    # Create search text for the encoder
    df['search_text'] = (
        "Title: " + df[col_title].fillna('') + ". " +
        "Category: " + df[col_cat].fillna('') + ". " +
        "Description: " + df[col_desc].fillna('')
    )
    return df, col_title, col_desc, col_cat

# --- 4. AI EXPLANATION FUNCTION (GEMINI) ---
def get_ai_explanation(query, results_df, api_key):
    if not api_key:
        return "⚠️ Please enter the Gemini API Key in the sidebar for AI summary."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        context = ""
        for i, row in results_df.iterrows():
            context += f"\n- {row['Title']}: {row['Description']}. Action: {row['Action Proposed']}\n"
        
        prompt = f"""
        You are a Senior QA/QC Engineering Expert.
        User Query: "{query}"
        Relevant Lessons Learned:
        {context}
        
        Provide a technical summary (4 lines) explaining the importance of these lessons 
        and the main preventive action the user should take. Use professional English.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# --- MAIN APP ---
st.title("🏗️ LLAA Intelligent Recommender")
st.markdown("### Smart Advisor for Engineering Lessons Learned")

# Load Resources
model_encoder = load_llm_encoder()
csv_file = "lecciones_aprendidas_calidad_600_v2.csv"

if os.path.exists(csv_file):
    df, c_title, c_desc, c_cat = load_data(csv_file)
    
    # Generate Embeddings (Once)
    with st.spinner("Indexing 600 unique technical records..."):
        corpus_embeddings = model_encoder.encode(df['search_text'].tolist(), convert_to_tensor=True)

    # --- SIDEBAR ---
    st.sidebar.header("Settings")
    location = st.sidebar.selectbox("Project Location:", ["Europe (EU)", "Rest of World (Non-EU)"])
    st.sidebar.divider()
    st.sidebar.write(f"Database Size: {len(df)} LLAA")

    # --- SEARCH ---
    query = st.text_input("Enter technical issue or scenario:", 
                         placeholder="e.g. Nitrogen purge loss during storage")

    if query:
        # Geo-filtering (CE MARKING)
        df_final = df.copy()
        valid_indices = list(range(len(df)))

        if location == "Rest of World (Non-EU)":
            mask = df[c_cat].str.contains('CE MARKING', case=False, na=False)
            df_final = df[~mask].reset_index(drop=True)
            valid_indices = df[~mask].index.tolist()

        # Semantic Match
        query_embedding = model_encoder.encode(query, convert_to_tensor=True)
        filtered_embeddings = corpus_embeddings[valid_indices]
        
        cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(3, len(df_final)))

        results_to_show = df_final.iloc[top_results.indices.tolist()]

        # --- AI SECTION ---
        st.info("### 🤖 Expert AI Guidance")
        with st.spinner("Gemini is synthesizing findings..."):
            explanation = get_ai_explanation(query, results_to_show, gemini_key)
            st.write(explanation)

        # --- RESULTS ---
        st.write("### 📂 Top Recommended Lessons")
        for i, (_, row) in enumerate(results_to_show.iterrows()):
            with st.expander(f"📌 {row[c_title]} (Match: {top_results.values[i]:.2f})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Ref:** {row['LL Ref']}")
                    st.write(f"**Category:** {row[c_cat]}")
                    st.write(f"**Project:** {row['Project']}")
                with col2:
                    st.write(f"**Description:** {row[c_desc]}")
                    st.success(f"**Action:** {row['Action Proposed']}")
                    st.warning(f"**Plan:** {row['Public Comments / Implementation Plan']}")
else:
    st.error("Dataset not found. Please run generate_dataset.py first.")