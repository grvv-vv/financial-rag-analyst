"""
Financial & SEC 10-K RAG Analyst - Streamlit Application
Interactive dashboard providing verifiable financial question answering
grounded in SEC 10-K disclosures with exact page and section citations.
"""

from pathlib import Path
import tempfile
import streamlit as st
import config
from src.rag_engine import FinancialRAGEngine

# Set up page styling and layout
st.set_page_config(
    page_title="Financial & SEC 10-K RAG Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber/FinTech dark theme styling
st.markdown(
    """
    <style>
    .reportview-container, .main {
        background-color: #0b0f19;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(0, 255, 65, 0.12);
        color: #00ff41;
        border: 1px solid rgba(0, 255, 65, 0.3);
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
        margin-right: 6px;
    }
    .citation-card {
        background: #131b2e;
        border-left: 3px solid #00ff41;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 13px;
    }
    .citation-header {
        font-weight: 700;
        color: #e6edf3;
        font-family: monospace;
        font-size: 12px;
        margin-bottom: 6px;
    }
    .stButton>button {
        background: #00ff41;
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 12px #00ff41;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_engine() -> FinancialRAGEngine:
    """Instantiates and caches the RAG orchestrator."""
    engine = FinancialRAGEngine()
    if config.SAMPLE_FILING_PATH.exists():
        engine.index_document(str(config.SAMPLE_FILING_PATH))
    return engine


engine = get_rag_engine()

# ===== SIDEBAR: CONFIGURATION & FILING MANAGEMENT =====
with st.sidebar:
    st.title("📈 SEC 10-K Pipeline")
    st.caption("Engineered by **Gaurav** (AI/ML Systems Engineer)")

    st.markdown("---")
    st.subheader("📁 Filing Ingestion")
    data_source = st.radio(
        "Select Document Source:",
        ["Sample SEC 10-K (Nova Dynamics 2025)", "Upload Custom Filing (PDF/TXT)"],
    )

    if data_source == "Upload Custom Filing (PDF/TXT)":
        uploaded_file = st.file_uploader("Upload SEC Filing", type=["pdf", "txt"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            if st.button("⚡ Index Uploaded Filing"):
                with st.spinner("Extracting text and generating vector embeddings..."):
                    res = engine.index_document(tmp_path)
                    st.success(f"Indexed {res['chunks_created']} chunks in {res['indexing_time_ms']}ms!")

    st.markdown("---")
    st.subheader("⚙️ Retrieval Hyperparameters")
    top_k = st.slider("Top-K Retrieved Context Chunks", min_value=1, max_value=5, value=config.DEFAULT_TOP_K)
    st.info("Embedding: `sentence-transformers/all-MiniLM-L6-v2` | Index: `ChromaDB` (Local Cosine)")

    st.markdown("---")
    st.subheader("💡 Suggested Financial Queries")
    q1 = "What was total consolidated revenue and operating profit in fiscal 2025?"
    q2 = "What are the primary semiconductor foundry supply chain risks?"
    q3 = "How much capital was allocated to AI R&D and future Capex guidance?"
    q4 = "What regulatory risks does the company face under the EU AI Act?"

    selected_preset = None
    if st.button("💵 Fiscal 2025 Revenue"):
        selected_preset = q1
    if st.button("⚠️ Foundry Supply Chain"):
        selected_preset = q2
    if st.button("🤖 AI R&D & Capex"):
        selected_preset = q3
    if st.button("🏛️ Regulatory & EU AI Act"):
        selected_preset = q4

# ===== MAIN CONTENT AREA =====
st.title("📈 Financial & SEC 10-K RAG Analyst")
st.markdown(
    "Production Retrieval-Augmented Generation engine for annual reports, SEC 10-K filings, and balance sheets. "
    "Features **verifiable page-level citations**, hallucination mitigation, and sub-second retrieval."
)

# Search Input
default_input = selected_preset or ""
query_text = st.text_input("Enter your financial inquiry:", value=default_input, placeholder="e.g. What were the total revenues and net income?")

col1, col2 = st.columns([1, 5])
with col1:
    search_triggered = st.button("⚡ Run RAG Analysis")

if search_triggered or (selected_preset and query_text):
    if not query_text.strip():
        st.warning("Please enter a question or select a suggested prompt.")
    else:
        with st.spinner("Executing dense vector retrieval and synthesizing analysis..."):
            result = engine.answer_query(query_text, top_k=top_k)

        # Telemetry Bar
        st.markdown(
            f'<div style="margin: 15px 0;">'
            f'<span class="metric-badge">⚡ Retrieval: {result["retrieval_time_ms"]} ms</span>'
            f'<span class="metric-badge">⏱️ Total: {result["total_time_ms"]} ms</span>'
            f'<span class="metric-badge">📄 Chunks Evaluated: {result["chunks_evaluated"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Synthesized Answer Card
        st.subheader("📋 Synthesized Financial Analysis")
        st.markdown(
            f"""
            <div style="background:#0f172a; border: 1px solid rgba(0,255,65,0.25); border-radius: 6px; padding: 18px; margin-bottom: 20px;">
                <div style="color: #f1f5f9; line-height: 1.7; font-size: 15px; white-space: pre-wrap;">{result['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Verifiable Citations
        st.subheader("📑 Verifiable Source Citations")
        if not result["citations"]:
            st.info("No citation sources evaluated.")
        else:
            for c in result["citations"]:
                with st.expander(f"Citation {c['citation_id']} • {c['section']} (Page {c['page']}) — Similarity: {c['similarity_score'] * 100:.1f}%"):
                    st.markdown(f"**Source Document:** `{c['source']}`")
                    st.markdown(f"**Filing Section:** `{c['section']}`")
                    st.markdown(f"**Page Attribution:** `Page {c['page']}`")
                    st.markdown("**Exact Extracted Context:**")
                    st.markdown(f"> *\"{c['snippet']}\"*")

st.markdown("---")
st.caption("Financial RAG Analyst • Architected by Gaurav • Open Source under MIT License")
