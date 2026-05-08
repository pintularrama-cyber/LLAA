import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="LLAA Intelligent Advisor", page_icon="🏗️", layout="wide")

# --- 1. AI CONFIGURATION (GEMINI) ---
gemini_key = None
if "GEMINI_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_KEY"]
else:
    gemini_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

# --- 2. SEMANTIC MODEL LOADING ---
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. DATA LOADING ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', engine='python')
        df.columns = df.columns.str.strip()
        # Create consolidated search column
        df['search_text'] = df['Title'] + " " + df['Description'] + " " + df['Knowledge Category']
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# --- 4. AI EXPLANATION ---
def get_ai_insight(query, results_df, api_key):
    if not api_key: return "⚠️ Please provide a Gemini API Key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        context = "\n".join([f"- {r['Title']}: {r['Description']}" for _, r in results_df.iterrows()])
        prompt = f"Expert QA/QC Engineer. User asks: {query}. Based on these lessons: {context}. Summarize in 4 lines why these are critical and the main action to take."
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Error: {e}"

# --- MAIN INTERFACE ---
st.title("🏗️ LLAA Intelligent Recommender")

encoder = load_encoder()
csv_path = "lecciones_aprendidas_calidad_600_v2.csv"

if os.path.exists(csv_path):
    df = load_data(csv_path)
    
    # Pre-compute embeddings
    with st.spinner("Indexing dataset..."):
        corpus_embeddings = encoder.encode(df['search_text'].tolist(), convert_to_tensor=True)

    # Sidebar Filter
    st.sidebar.header("Filter Settings")
    location = st.sidebar.selectbox("Project Region:", ["Europe (EU)", "Rest of World (Non-EU)"])
    region_code = "EU" if location == "Europe (EU)" else "Non-EU"

    # Search Bar
    query = st.text_input("Describe your quality issue (e.g., welding cracking or painting delamination):")

    if query:
        # STRICT REGIONAL FILTERING
        mask = df['Region'] == region_code
        filtered_df = df[mask].reset_index(drop=True)
        filtered_embeddings = corpus_embeddings[df[mask].index.tolist()]

       # SEMANTIC SEARCH
        query_emb = encoder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, filtered_embeddings)[0]
        
        # Pedimos 10 resultados en lugar de 3 para tener margen de filtrado
        top_k_raw = torch.topk(scores, k=min(10, len(filtered_df)))
        
        # Filtramos para que no haya títulos repetidos en el top 3 final
        raw_results = filtered_df.iloc[top_k_raw.indices.tolist()]
        raw_results['score'] = top_k_raw.values.tolist()
        
        # Eliminamos filas que tengan el mismo Título o Descripción (mantenemos la de mayor score)
        recommendations = raw_results.drop_duplicates(subset=['Title']).head(3)

        # DISPLAY AI INSIGHT
        st.info("### 🤖 AI Expert Summary")
        st.write(get_ai_insight(query, recommendations, gemini_key))

        # DISPLAY INDIVIDUAL LESSONS
        st.write("### 📂 Top Matching Lessons Learned")
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with st.expander(f"📌 {row['Title']} (Score: {top_k.values[i]:.2f})"):
                st.write(f"**Project:** {row['Project']} | **Category:** {row['Knowledge Category']}")
                st.write(f"**Description:** {row['Description']}")
                st.success(f"**Action:** {row['Action Proposed']}")

else:
    st.warning("Please run 'generate_dataset.py' to create the technical database.")