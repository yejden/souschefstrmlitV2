import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SousChef 🍳",
    page_icon="🍳",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #FDFAF5;
    color: #2C2C2C;
}

.block-container {
    padding-top: 2rem;
    max-width: 760px;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #B5451B;
    margin-bottom: 0;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1.1rem;
    color: #7A7A7A;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
}

/* Recipe card */
.recipe-card {
    background: #FFFFFF;
    border-left: 5px solid #B5451B;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

.recipe-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: #B5451B;
    margin-bottom: 0.3rem;
}

.recipe-desc {
    color: #555;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.section-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #B5451B;
    font-weight: 700;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

.ingredient-pill {
    display: inline-block;
    background: #FFF0EB;
    border: 1px solid #F4C5B3;
    color: #8B2E0F;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    margin: 0.2rem;
    font-size: 0.88rem;
}

.step-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1rem;
    gap: 1rem;
}

.step-num {
    background: #B5451B;
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    min-width: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
}

.step-content {
    flex: 1;
}

.step-desc {
    font-size: 0.97rem;
    color: #FFFFFF;
    line-height: 1.5;
}

.step-time {
    font-size: 0.78rem;
    color: #CCC;
    margin-top: 0.2rem;
}

.time-badge {
    background: #FFF0EB;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    color: #B5451B;
    font-weight: 600;
}

/* Input */
.stTextInput > div > div > input {
    border: 2px solid #E8D8CF !important;
    border-radius: 8px !important;
    background: #FFFAF7 !important;
    color: #1A1A1A !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #B5451B !important;
    box-shadow: 0 0 0 2px rgba(181,69,27,0.15) !important;
    color: #1A1A1A !important;
}

.stTextInput > div > div > input::placeholder {
    color: #999 !important;
}

/* Button */
.stButton > button {
    background: #B5451B !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    transition: background 0.2s ease !important;
}

.stButton > button:hover {
    background: #8B2E0F !important;
}

/* API key input */
.stTextInput[data-testid="stTextInput-apikey"] input {
    font-family: monospace !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class StapSchema(BaseModel):
    stap: str = Field(description="Beschrijving van de stap")
    tijd: str = Field(description="Geschatte tijd voor deze stap")

class RecipeSchema(BaseModel):
    titel: str = Field(description="Titel of naam van het recept")
    beschrijving: str = Field(description="Beschrijving van recept")
    ingredienten: list[str] = Field(description="Lijst van ingredienten uit het recept")
    stappen: list[StapSchema] = Field(description="Stappen in het recept met geschatte tijd")

# ── Prompt ────────────────────────────────────────────────────────────────────
RETRIEVE_PROMPT = """
You are a professional chef assistant.

Generate 10 possible recipes internally.
Randomly select ONE.
Return ONLY the recipe in valid JSON format.

Do not include markdown, code fences, explanations, or extra text.

User request:
{user_input}

### Output Format (JSON)
{{
  "titel": "naam van het recept",
  "beschrijving": "de beschrijving van het gerecht",
  "ingredienten": [
  "ingredient 1", 
  "ingredient 2", 
  "ingredient 3"
  ],
  "stappen": [
  {{
  "stap": "beschrijving van stap", 
  "tijd": "geschatte tijd"
  }}, 
  {{"stap": 
  "beschrijving van stap", 
  "tijd": "geschatte tijd"
  }}
  ]
}}
"""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">SousChef 🍳</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Jouw persoonlijke chef-assistent — vertel wat je wilt koken.</p>', unsafe_allow_html=True)

# ── API Key input ─────────────────────────────────────────────────────────────
with st.expander("⚙️ Groq API Key instellen", expanded=False):
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.get("groq_api_key", ""),
        type="password",
        placeholder="gsk_...",
        key="stTextInput-apikey"
    )
    if api_key_input:
        st.session_state["groq_api_key"] = api_key_input
        st.success("API key opgeslagen ✓")

# Fall back to hardcoded key if not set via UI
groq_api_key = st.session_state.get("groq_api_key", "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")

# ── Main input ────────────────────────────────────────────────────────────────
st.markdown("---")
user_input = st.text_input(
    "Wat wil je koken?",
    placeholder="bijv. een makkelijke pasta, Aziatische soep, gezond ontbijt...",
    label_visibility="visible"
)

col1, col2 = st.columns([1, 3])
with col1:
    generate = st.button("Recept ophalen →")

# ── Generate recipe ───────────────────────────────────────────────────────────
if generate:
    if not user_input.strip():
        st.warning("Vul eerst in wat je wilt koken! 🍽️")
    elif not groq_api_key:
        st.error("Stel eerst een Groq API key in via de instellingen hierboven.")
    else:
        with st.spinner("Even nadenken... 🧑‍🍳"):
            try:
                os.environ["GROQ_API_KEY"] = groq_api_key
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
                structured_llm = llm.with_structured_output(RecipeSchema, method="json_mode")
                prompt = RETRIEVE_PROMPT.format(user_input=user_input)
                recipe: RecipeSchema = structured_llm.invoke(prompt)
                st.session_state["recipe"] = recipe
            except Exception as e:
                st.error(f"Er ging iets mis: {e}")

# ── Display recipe ────────────────────────────────────────────────────────────
if "recipe" in st.session_state:
    r: RecipeSchema = st.session_state["recipe"]

    # Title + description
    st.markdown(f"""
    <div class="recipe-card">
        <div class="recipe-title">{r.titel}</div>
        <div class="recipe-desc">{r.beschrijving}</div>
    </div>
    """, unsafe_allow_html=True)

    # Ingredients
    st.markdown('<div class="section-label">🥕 Ingrediënten</div>', unsafe_allow_html=True)
    pills_html = "".join(f'<span class="ingredient-pill">{i}</span>' for i in r.ingredienten)
    st.markdown(f"<div style='margin-bottom:1.5rem'>{pills_html}</div>", unsafe_allow_html=True)

    # Steps
    st.markdown('<div class="section-label">👨‍🍳 Bereidingswijze</div>', unsafe_allow_html=True)
    steps_html = ""
    for idx, stap in enumerate(r.stappen, 1):
        steps_html += f"""
        <div class="step-row">
            <div class="step-num">{idx}</div>
            <div class="step-content">
                <div class="step-desc">{stap.stap}</div>
                <div class="step-time">⏱ <span class="time-badge">{stap.tijd}</span></div>
            </div>
        </div>
        """
    st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Ander recept genereren"):
        del st.session_state["recipe"]
        st.rerun()
