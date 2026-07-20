"""Streamlit demo — Phase 4 (placeholder).

Run once the model exists:
    pip install -e ".[app,nlp]"
    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="ColombiaCheck Verdict Classifier", page_icon="🔎")

st.title("🔎 ColombiaCheck Verdict Classifier")
st.caption(
    "Demo educativa. El modelo predice **cómo ColombiaCheck etiquetaría** una "
    "afirmación — NO si es objetivamente verdadera o falsa. "
    "Ver Model Card y Data Statement."
)

st.info("Modelo aún no entrenado (Fase 3 pendiente). Esta es la interfaz base.")

claim = st.text_area("Afirmación a clasificar", placeholder="Escribe una afirmación…")
if st.button("Clasificar", disabled=not claim):
    st.warning("Inferencia no disponible todavía: entrena el modelo en la Fase 3.")
    # TODO Phase 3: load fine-tuned BETO, predict, show label + probabilities.
