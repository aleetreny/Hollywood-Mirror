"""
Hollywood Mirror — App Streamlit.
Input: st.text_area con guion o idea.
Output: Top 5 películas similares, afinidad %, punto en la galaxia.
"""
import streamlit as st

st.set_page_config(page_title="Hollywood Mirror", layout="wide")
st.title("Hollywood Mirror")
st.caption("Pega un fragmento de guion o una idea para encontrar películas con estilo similar.")
st.text_area("Tu texto", placeholder="Escribe o pega aquí...", height=200)
# TODO: inferencia con embeddings, similitud coseno, visualización galaxia
