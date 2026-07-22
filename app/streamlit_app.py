"""Streamlit demo — ColombiaCheck verdict classifier (fine-tuned BETO).

Loads the published model straight from the Hugging Face Hub
(`polux89/beto-colombiacheck`), so the app needs no local checkpoint.

Run:
    pip install -e ".[app]"
    streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

# Semver of the deployed demo — keep in sync with `version` in pyproject.toml.
APP_VERSION = "0.1.0"

HUB_MODEL = "polux89/beto-colombiacheck"
GITHUB_URL = "https://github.com/POLUX89/NLP-Fake-News-Colombia"
HUB_URL = f"https://huggingface.co/{HUB_MODEL}"

LABEL_ICON = {"Falso": "🔴", "Cuestionable": "🟠", "Verdadero": "🟢"}
EXAMPLES = [
    "Una persona fallecida fue jurado de votación en las elecciones de 2026",
    "Video de las disidencias de las Farc en la Alcaldía de Tibú",
    "La favorabilidad del presidente subió al 82% según la última encuesta",
]

st.set_page_config(page_title="ColombiaCheck Verdict Classifier", page_icon="🔎")


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Download (once) and cache the fine-tuned model from the HF Hub."""
    from transformers import pipeline

    return pipeline("text-classification", model=HUB_MODEL, top_k=None)


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(
        f"[BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) "
        f"fine-tuneado sobre 2.935 chequeos de "
        f"[ColombiaCheck](https://colombiacheck.com) → "
        f"[`{HUB_MODEL}`]({HUB_URL})."
    )
    st.markdown("**Métricas (test, n=439)**")
    st.table(
        pd.DataFrame(
            {"F1": [0.806, 0.410, 0.000, 0.405]},
            index=["Falso", "Cuestionable", "Verdadero", "macro-F1"],
        )
    )
    st.caption(
        "macro-F1 95% CI: [0.371, 0.440]. `Verdadero` no es aprendible con 93 "
        "ejemplos (escasez estructural, documentada en el Model Card)."
    )
    st.markdown(f"[📂 Repositorio]({GITHUB_URL}) · [🤗 Model card]({HUB_URL})")
    st.caption(
        f"Demo v{APP_VERSION} · Los textos de los chequeos son propiedad "
        "de ColombiaCheck."
    )

# --------------------------------------------------------------------- main
st.title("🔎 ColombiaCheck Verdict Classifier")
st.warning(
    "**Esto NO es un detector de verdad.** El modelo predice **cómo "
    "ColombiaCheck etiquetaría** una afirmación (Falso / Cuestionable / "
    "Verdadero), aprendido de los chequeos de una única organización. "
    "En test nunca acierta la clase `Verdadero` (F1 = 0.0). "
    "Es una demo educativa — no la uses para decidir sobre personas, "
    "medios ni publicaciones."
)

if "claim" not in st.session_state:
    st.session_state.claim = ""


def _use_example() -> None:
    st.session_state.claim = st.session_state.example or ""


st.selectbox(
    "O prueba un ejemplo",
    EXAMPLES,
    index=None,
    key="example",
    placeholder="Elegir una afirmación de ejemplo…",
    on_change=_use_example,
)

claim = st.text_area(
    "Afirmación a clasificar (en español)",
    key="claim",
    placeholder="Escribe una afirmación corta, p. ej. «El ministro anunció…»",
    height=100,
)

if st.button("Clasificar", type="primary", disabled=not claim.strip()):
    with st.spinner("Clasificando… (la primera vez descarga el modelo, ~420 MB)"):
        clf = load_classifier()
        scores = clf([claim.strip()])[0]  # top_k=None -> las 3 clases con score

    best = max(scores, key=lambda s: s["score"])
    st.subheader(
        f"{LABEL_ICON.get(best['label'], '⚪')} {best['label']} "
        f"({best['score']:.1%})"
    )

    probs = (
        pd.DataFrame(scores)
        .rename(columns={"label": "Clase", "score": "Probabilidad"})
        .set_index("Clase")
        .reindex(["Falso", "Cuestionable", "Verdadero"])
    )
    st.bar_chart(probs, horizontal=True)

    st.caption(
        "Probabilidades del modelo, no veredictos. Un chequeo real requiere "
        "verificación humana — consulta colombiacheck.com."
    )
