import os
import sys
import json
import re
import streamlit as st
import pandas as pd
import PyPDF2
from io import BytesIO

# Add parent directory to path to import main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the NLP pipeline
try:
    from main import NLPOrchestrator
except ImportError:
    st.error("Could not import main.py. Make sure it exists in the parent directory.")
    st.stop()

st.set_page_config(
    page_title="Local LLM Text Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
body { background-color: #0E1117; color: #FAFAFA; }
.stApp { background-color: #0E1117; }

h1, h2, h3 { color: #E6E8EE; font-family: 'Inter', sans-serif; }

.block {
    background: #1E2330;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #2B313E;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.entity {
    background: linear-gradient(90deg, #1B2A4A 0%, #243B66 100%);
    color: #9DBBFF;
    padding: 6px 14px;
    margin: 4px;
    border-radius: 999px;
    display: inline-block;
    font-weight: 500;
    font-size: 13px;
    border: 1px solid #35476B;
}

.stButton>button {
    background-color: #4F8CFF;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.2s;
}

.stButton>button:hover {
    background-color: #3A76E8;
    box-shadow: 0 4px 12px rgba(79, 140, 255, 0.3);
}

.stTextArea textarea {
    background-color: #1E2330;
    border-color: #2B313E;
    color: #FAFAFA;
}
</style>
""", unsafe_allow_html=True)

# Cache the NLP model to avoid reloading
@st.cache_resource
def get_nlp_orchestrator():
    return NLPOrchestrator()

def extract_text_from_pdf(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    st.info("Using local Ollama instance with `main.py` pipeline.")
    
    st.markdown("### Model Status")
    try:
        nlp = get_nlp_orchestrator()
        st.success(f"Model Loaded: {nlp.model_id}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Main Content
st.title("🧠 Local LLM Text Intelligence")
st.markdown("Upload a document or paste text to perform advanced NLP analysis using Llama 3.1.")

st.markdown("<div class='block'>", unsafe_allow_html=True)
st.subheader("Input")

uploaded_file = st.file_uploader(
    "Upload file (TXT, CSV, PDF)",
    type=["txt", "csv", "pdf"],
    accept_multiple_files=False
)

input_text = ""

if uploaded_file:
    try:
        if uploaded_file.type == "text/plain":
            input_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            input_text = "\n".join(df.astype(str).values.flatten())
        elif uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                input_text = extract_text_from_pdf(uploaded_file.read())
        
        st.info(f"File '{uploaded_file.name}' loaded successfully. ({len(input_text)} chars)")
    except Exception as e:
        st.error(f"Error loading file: {e}")

input_text = st.text_area(
    "Text to Analyze",
    value=input_text,
    height=280,
    max_chars=10000,
    placeholder="Paste text or upload a file..."
)

analyze_btn = st.button("🚀 Analyze Text")
st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn:
    if not input_text.strip():
        st.error("Please provide some text to analyze.")
    else:
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        try:
            with st.spinner("Running inference pipeline... this might take a moment"):
                # Clear the LLM cache to ensure fresh results for new input
                nlp._llm_cache.clear()
                # Call the process method from NLPOrchestrator
                result = nlp.process(input_text)
            
            # Display Result
            st.success("Analysis Complete!")
            
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Clean & Translate", "🧾 Raw JSON"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Language", result.get("lang_det", "Unknown"), f"{result.get('lang_det_confidence_score', 0)*100:.0f}% Conf")
                    st.metric("Domain", result.get("domain_ident", "General"), f"{result.get('domain_ident_confidence_score', 0)*100:.0f}% Conf")
                
                with col2:
                    st.metric("Sentiment", result.get("sentiment", "Neutral"), f"{result.get('sentiment_confidence_score', 0)*100:.0f}% Conf")
                    st.metric("Country", result.get("Country_iden", "Unknown"))

                st.markdown("### Summary")
                st.info(result.get("Summary", "No summary available."))

                st.markdown("### Named Entities")
                ner = result.get("NER", {})
                if not any(ner.values()) or (isinstance(ner, dict) and not any(ner.values())):
                     st.write("No entities detected.")
                else:
                    for label, values in ner.items():
                        if values:
                             # values might be a comma separated string or list, main.py tostr makes it a string
                             # Splitting by comma for display if it looks like a list
                            vals = [v.strip() for v in str(values).split(',')] if ',' in str(values) else [str(values)]
                            st.markdown(f"**{label}**")
                            for v in vals:
                                if v:
                                    st.markdown(f"<span class='entity'>{v}</span>", unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Original Cleaned")
                st.text_area("Cleaned", result.get("Cleaned_content", ""), height=150)
                
                st.subheader("Translation (to English)")
                st.text_area("Translated", result.get("Translation", ""), height=150)

            with tab3:
                st.json(result)
                st.download_button(
                    "Download JSON Response",
                    json.dumps(result, indent=2, ensure_ascii=False),
                    file_name="inference_result.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"An error occurred during verification: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        st.markdown("</div>", unsafe_allow_html=True)
