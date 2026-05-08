import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="QA/QC LLAA Advisor", page_icon="🏗️", layout="wide")

# --- 1. CONFIGURACIÓN DE SEGURIDAD (GEMINI) ---
# Intentamos leer de secrets, si no, pedimos por el sidebar
gemini_key = None
if "GEMINI_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_KEY"]
else:
    gemini_key = st.sidebar.text_input("Introduce Gemini API Key:", type="password")

# --- 2. CARGA DE MODELO SEMÁNTICO (CACHE) ---
@st.cache_resource
def load_llm_encoder():
    # Modelo ligero y potente para Mac
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. CARGA DE DATOS ROBUSTA ---
@st.cache_data
def load_data(file_path):
    # Intentamos detectar el separador (coma o punto y coma)
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return None

    # Limpiamos espacios en los nombres de las columnas
    df.columns = df.columns.str.strip()
    
    # Nombres de columnas esperados (basado en tu contexto)
    # Si tus columnas reales varían un poco, esto las busca de forma flexible
    col_title = 'Title' if 'Title' in df.columns else df.columns[1]
    col_desc = 'Description' if 'Description' in df.columns else df.columns[6]
    col_cat = 'Knowledge Category' if 'Knowledge Category' in df.columns else df.columns[5]

    # Creamos el texto para el buscador
    df['search_text'] = (
        "Título: " + df[col_title].fillna('') + ". " +
        "Categoría: " + df[col_cat].fillna('') + ". " +
        "Descripción: " + df[col_desc].fillna('')
    )
    return df, col_title, col_desc, col_cat

# --- 4. FUNCIÓN DE EXPLICACIÓN (GEMINI) ---
def get_ai_explanation(query, results_df, api_key):
    if not api_key:
        return "⚠️ Introduce la API Key en el sidebar para obtener la síntesis de la IA."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        contexto = ""
        for i, row in results_df.iterrows():
            contexto += f"\n- {row['Title']}: {row['Description']}. Acción: {row['Action Proposed']}\n"
        
        prompt = f"""
        Eres un experto Senior en Ingeniería y Calidad (QA/QC).
        El usuario pregunta: "{query}"
        Basado en estas lecciones aprendidas:
        {contexto}
        
        Explica en un párrafo técnico de 4 líneas por qué estas lecciones son relevantes y qué acción principal debe tomar el usuario.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error en Gemini: {e}"

# --- MAIN APP ---
st.title("🏗️ LLAA Intelligent Recommender")
st.markdown("### Motor de Recomendación de Lecciones Aprendidas para Ingeniería")

# Carga de recursos
model_encoder = load_llm_encoder()
csv_file = "lecciones_aprendidas_calidad_600_v2.csv"

if os.path.exists(csv_file):
    df, c_title, c_desc, c_cat = load_data(csv_file)
    
    # Generar Embeddings (Solo una vez)
    with st.spinner("Indexando lecciones semánticamente..."):
        corpus_embeddings = model_encoder.encode(df['search_text'].tolist(), convert_to_tensor=True)

    # --- SIDEBAR FILTROS ---
    st.sidebar.header("Filtros de Proyecto")
    ubicacion = st.sidebar.selectbox("Ubicación del Proyecto:", ["Europa (EU)", "Resto del Mundo (Non-EU)"])
    
    st.sidebar.divider()
    st.sidebar.write(f"Total registros: {len(df)}")

    # --- BÚSQUEDA ---
    query = st.text_input("Describe el escenario o problema de calidad:", placeholder="Ej: Corrosión en estructuras metálicas offshore")

    if query:
        # Aplicar restricción geográfica (CE MARKING)
        df_final = df.copy()
        indices_validos = list(range(len(df)))

        if ubicacion == "Resto del Mundo (Non-EU)":
            # Filtramos lecciones que tengan 'CE MARKING' en la categoría
            mask = df[c_cat].str.contains('CE MARKING', case=False, na=False)
            df_final = df[~mask].reset_index(drop=True)
            indices_validos = df[~mask].index.tolist()

        # Similitud Semántica
        query_embedding = model_encoder.encode(query, convert_to_tensor=True)
        # Usamos solo los embeddings de los registros que pasaron el filtro geográfico
        embeddings_filtrados = corpus_embeddings[indices_validos]
        
        cos_scores = util.cos_sim(query_embedding, embeddings_filtrados)[0]
        top_results = torch.topk(cos_scores, k=min(3, len(df_final)))

        # Resultados para mostrar e informar a Gemini
        results_to_show = df_final.iloc[top_results.indices.tolist()]

        # --- SECCIÓN IA ---
        st.info("### 🤖 Recomendación del Experto IA")
        with st.spinner("Gemini analizando resultados..."):
            explicacion = get_ai_explanation(query, results_to_show, gemini_key)
            st.write(explicacion)

        # --- TABLA DE RESULTADOS ---
        st.write("### 📂 Lecciones Técnicas Relacionadas")
        for i, (_, row) in enumerate(results_to_show.iterrows()):
            with st.expander(f"📌 {row[c_title]} (Score: {top_results.values[i]:.2f})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Ref:** {row.get('LL Ref', 'N/A')}")
                    st.write(f"**Categoría:** {row[c_cat]}")
                    st.write(f"**Status:** {row.get('Status', 'N/A')}")
                with col2:
                    st.write(f"**Descripción:** {row[c_desc]}")
                    st.success(f"**Acción Propuesta:** {row.get('Action Proposed', 'N/A')}")
                    if 'Public Comments / Implementation Plan' in row:
                        st.warning(f"**Plan de Acción:** {row['Public Comments / Implementation Plan']}")

else:
    st.error(f"No se encuentra el archivo {csv_file} en la carpeta del proyecto.")