import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai  # <--- Nueva librería

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="QA/QC LLAA Advisor", page_icon="🏗️")

# --- 1. CONFIGURACIÓN DE GEMINI ---
st.sidebar.title("Configuración AI")
gemini_key = st.secrets["GEMINI_KEY"]
def generate_explanation(query, results_df, api_key):
    if not api_key:
        return "⚠️ Por favor, introduce la API Key de Gemini para ver la explicación."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Versión rápida y eficiente
    
    # Construimos el contexto para el LLM
    contexto_lecciones = ""
    for i, row in results_df.iterrows():
        contexto_lecciones += f"\n- LECCIÓN {i+1} ({row['Title']}): {row['Description']}. Acción propuesta: {row['Action Proposed']}\n"
    
    prompt = f"""
    Eres un experto Senior en QA/QC de ingeniería. 
    El usuario tiene el siguiente problema o duda: "{query}"
    
    He encontrado las siguientes lecciones aprendidas relevantes en nuestra base de datos:
    {contexto_lecciones}
    
    Tu tarea es redactar un breve párrafo (máximo 4-5 líneas) explicando al usuario por qué estas lecciones 
    específicas son críticas para su caso y qué debería priorizar para evitar fallos de calidad. 
    Sé profesional, técnico y directo.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al conectar con Gemini: {e}"

# --- 2. (Mantenemos las funciones de carga de datos y embeddings de antes) ---
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_prep_data(file_path):
    df = pd.read_csv(file_path)
    df['search_text'] = "Título: " + df['Title'].fillna('') + ". Descripción: " + df['Description'].fillna('')
    return df

@st.cache_data
def get_embeddings(_model, _texts):
    return _model.encode(_texts, convert_to_tensor=True)

# Inicializar
model_st = get_model()
df = load_and_prep_data("lecciones_aprendidas_calidad_600_v2.csv")
corpus_embeddings = get_embeddings(model_st, df['search_text'].tolist())

# --- INTERFAZ ---
st.title("🏗️ Motor de Recomendación Inteligente LLAA")

ubicacion = st.radio("Ubicación del Proyecto:", ["Europa (EU)", "Resto del Mundo (Non-EU)"])

query = st.text_input("Describe tu problema de calidad:")

if query:
    # --- FILTRADO ---
    df_filtrado = df.copy()
    emb_filtrados = corpus_embeddings
    
    if ubicacion == "Resto del Mundo (Non-EU)":
        mask_ce = df_filtrado['Knowledge Category'].str.contains('CE MARKING', case=False, na=False)
        df_filtrado = df_filtrado[~mask_ce].reset_index(drop=True)
        emb_filtrados = corpus_embeddings[df[~mask_ce].index.tolist()]

    # --- BÚSQUEDA ---
    query_emb = model_st.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, emb_filtrados)[0]
    top_results = torch.topk(cos_scores, k=min(3, len(df_filtrado)))
    
    # Guardamos los resultados en un DF temporal para el LLM
    indices = [idx.item() for idx in top_results.indices]
    recomendaciones_df = df_filtrado.iloc[indices]

    # --- RESULTADOS ---
    st.write("### 🤖 Razonamiento del Experto (Gemini):")
    with st.spinner("Analizando lecciones..."):
        explicacion = generate_explanation(query, recomendaciones_df, gemini_key)
        st.info(explicacion)

    st.write("### 📚 Detalle de Lecciones Encontradas:")
    for _, leccion in recomendaciones_df.iterrows():
        with st.expander(f"📌 {leccion['Title']}"):
            st.write(f"**Descripción:** {leccion['Description']}")
            st.write(f"**Acción:** {leccion['Action Proposed']}")