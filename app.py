import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="QA/QC LLAA Advisor", page_icon="🏗️", layout="wide")

# --- 1. SECURITY CONFIGURATION (GEMINI) ---
# Trying to read from secrets, otherwise request via sidebar
gemini_key = None
if "GEMINI_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_KEY"]
else:
    gemini_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

# --- 2. SEMANTIC MODEL LOADING (CACHE) ---
@st.cache_resource
def load_llm_encoder():
    # Multilingual model: Works perfectly with English, Spanish, and 50+ languages
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 3. ROBUST DATA LOADING ---
@st.cache_data
def load_data(file_path):
    # Automatically detect separator (comma or semicolon)
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Clean whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Identify key columns (flexible search)
    col_title = 'Title' if 'Title' in df.columns else df.columns[1]
    col_desc = 'Description' if 'Description' in df.columns else df.columns[6]
    col_cat = 'Knowledge Category' if 'Knowledge Category' in df.columns else df.columns[5]

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
        return "⚠️ Please enter the Gemini API Key in the sidebar to get AI insights."
    
    try:
        genai.configure(api_key=api_key)
        # Using the full model path to avoid 404 errors
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        context = ""
        for i, row in results_df.iterrows():
            context += f"\n- {row['Title']}: {row['Description']}. Proposed Action: {row['Action Proposed']}\n"
        
        prompt = f"""
        You are a Senior Engineering QA/QC Expert.
        The user is asking: "{query}"
        Based on the following Lessons Learned found in our database:
        {context}
        
        Provide a technical summary (max 4-5 lines) explaining why these specific lessons are relevant 
        to the user's case and what primary action they should prioritize to ensure quality. 
        Answer in professional English.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# --- MAIN APP ---
st.title("🏗️ LLAA Intelligent Recommender")
st.markdown("### Smart Recommendation Engine for Engineering Lessons Learned")

# Load Resources
model_encoder = load_llm_encoder()
csv_file = "lecciones_aprendidas_calidad_600_v2.csv"

if os.path.exists(csv_file):
    df, c_title, c_desc, c_cat = load_data(csv_file)
    
    # Generate Embeddings (Once)
    with st.spinner("Indexing lessons semantically..."):
        corpus_embeddings = model_encoder.encode(df['search_text'].tolist(), convert_to_tensor=True)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Project Configuration")
    location = st.sidebar.selectbox("Project Location:", ["Europe (EU)", "Rest of World (Non-EU)"])
    
    st.sidebar.divider()
    st.sidebar.write(f"Total Records: {len(df)}")
    st.sidebar.info("This engine uses semantic search. It understands concepts even if keywords don't match exactly.")

    # --- SEARCH INTERFACE ---
    query = st.text_input("Describe your quality scenario or technical issue:", 
                         placeholder="e.g., Corrosion in offshore metallic structures under high salinity")

    if query:
        # Apply geographic restriction (CE MARKING)
        df_final = df.copy()
        valid_indices = list(range(len(df)))

        if location == "Rest of World (Non-EU)":
            # Filter out lessons containing 'CE MARKING' in the category
            mask = df[c_cat].str.contains('CE MARKING', case=False, na=False)
            df_final = df[~mask].reset_index(drop=True)
            valid_indices = df[~mask].index.tolist()

        # Semantic Similarity
        query_embedding = model_encoder.encode(query, convert_to_tensor=True)
        # Filter embeddings to match the geographic selection
        filtered_embeddings = corpus_embeddings[valid_indices]
        
        cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(3, len(df_final)))

        # Collect results for UI and AI
        results_to_show = df_final.iloc[top_results.indices.tolist()]

        # --- AI SECTION ---
        st.info("### 🤖 AI Expert Insight")
        with st.spinner("Gemini is analyzing the results..."):
            explanation = get_ai_explanation(query, results_to_show, gemini_key)
            st.write(explanation)

        # --- RESULTS TABLE ---
        st.write("### 📂 Related Technical Lessons")
        for i, (_, row) in enumerate(results_to_show.iterrows()):
            with st.expander(f"📌 {row[c_title]} (Relevance Score: {top_results.values[i]:.2f})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Ref:** {row.get('LL Ref', 'N/A')}")
                    st.write(f"**Category:** {row[c_cat]}")
                    st.write(f"**Status:** {row.get('Status', 'N/A')}")
                with col2:
                    st.write(f"**Description:** {row[c_desc]}")
                    st.success(f"**Proposed Action:** {row.get('Action Proposed', 'N/A')}")
                    if 'Public Comments / Implementation Plan' in row:
                        st.warning(f"**Implementation Plan:** {row['Public Comments / Implementation Plan']}")

else:
    st.error(f"File {csv_file} not found in the project root directory.")