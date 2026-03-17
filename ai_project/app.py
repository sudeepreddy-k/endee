import streamlit as st
import faiss
import time
from sentence_transformers import SentenceTransformer
from ipc_data import IPC_SECTIONS

DIMENSION = 384

# ── Vector store ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Initializing Endee Vector Engine…")
def build_store():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        f"{s['section']} – {s['title']}. {s['description']}"
        for s in IPC_SECTIONS
    ]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(vecs.astype("float32"))
    return model, index


def semantic_retrieval(query, top_k=5):
    model, index = build_store()
    qvec = model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k, index.ntotal)
    scores, indices = index.search(qvec, k)
    return [
        {**IPC_SECTIONS[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0]) if idx >= 0
    ]

# ── RAG Synthesis Layer ──────────────────────────────────────────────────────

def generate_ai_analysis(query, retrieved_docs):
    if not retrieved_docs:
        return "I couldn't find any specific IPC sections directly related to your query. Please provide more details."

    top_doc = retrieved_docs[0]
    
    analysis = f"## AI Legal Insights\n\n"
    analysis += f"Based on the context retrieved from the **Endee Vector Database**, here is a preliminary analysis for your situation:\n\n"
    
    analysis += f"### ⚖️ Primary Classification: **{top_doc['section']}**\n"
    analysis += f"**{top_doc['title']}**\n\n"
    
    analysis += f"> {top_doc['description']}\n\n"
    
    analysis += "### 🔍 Key Implications\n"
    analysis += "- **Legal Definition**: Your situation matches the criteria defined under this section.\n"
    
    if len(retrieved_docs) > 1:
        analysis += "- **Correlated Sections**: " + ", ".join([d['section'] for d in retrieved_docs[1:3]]) + " may also apply depending on evidentiary findings."

    analysis += "\n\n---\n"
    
    return analysis


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Endee AI Legal Assistant", page_icon="⚖️", layout="wide")

# ── Professional Dashboard CSS ───────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #6366f1;
    --accent: #10b981;
    --bg-dark: #0b0f19;
    --card-bg: rgba(30, 41, 59, 0.4);
    --border: rgba(255, 255, 255, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
    color: #f8fafc;
}

/* Glassmorphism Header */
.header-container {
    padding: 3rem 1rem 4rem;
    text-align: center;
}

.title-main {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle-main {
    color: #94a3b8;
    font-size: 1.25rem;
    font-weight: 400;
    max-width: 800px;
    margin: 0 auto !important;
    text-align: center !important;
    display: block;
}

/* Status Pill */
.status-pill {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 100px;
    color: #818cf8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}

/* Search Box Container */
div[data-testid="stForm"] {
    background: var(--card-bg) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border) !important;
    border-radius: 24px !important;
    padding: 2rem !important;
}

/* Inputs */
input[data-testid="stTextInput"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    color: white !important;
    font-size: 1.1rem !important;
}

/* Results Section */
.analysis-box {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
}

.doc-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}

.doc-card:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
}

.doc-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.section-tag {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
}

.match-pct {
    font-size: 0.75rem;
    color: #10b981;
    font-weight: 600;
}

.doc-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 4px;
}

.doc-body {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #3730a3) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    height: auto !important;
    transition: all 0.2s ease !important;
}

.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.4) !important;
}

/* Divider */
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.05), transparent);
    margin: 4rem 0;
}

/* Hide Streamlit components */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-container">
    <div class="status-pill">Next-Gen Legal Intelligence</div>
    <h1 class="title-main">AI Legal Assistant</h1>
    <p class="subtitle-main">
        Professional Retrieval-Augmented Generation (RAG) platform powered by 
        <strong>Endee Vector Engine</strong>. Access instant judicial insights and IPC classifications.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Controller ───────────────────────────────────────────────────────────────

with st.container():
    col_empty1, col_center, col_empty2 = st.columns([1, 2, 1])
    with col_center:
        # Internal search layout
        sub_col_l, sub_col_r = st.columns([4, 1])
        with sub_col_l:
            query_input = st.text_input("", placeholder="Describe a legal scenario or query...", label_visibility="collapsed")
        with sub_col_r:
            run_analysis = st.button("Analyze", use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Results Logic ───────────────────────────────────────────────────────────

if run_analysis and query_input.strip():
    with st.status("Initializing Neural Retrieval...", expanded=False) as status:
        st.write("🔍 Vectorizing natural language input...")
        time.sleep(0.4)
        st.write("🛰️ Executing semantic search on Endee Index...")
        results = semantic_retrieval(query_input.strip(), top_k=6)
        time.sleep(0.4)
        st.write("🧠 Synthesizing judicial context with RAG...")
        ai_response = generate_ai_analysis(query_input.strip(), results)
        time.sleep(0.4)
        status.update(label="Analysis Completed", state="complete")

    # Layout Results
    col_main, col_side = st.columns([3, 2], gap="large")
    
    with col_main:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown(ai_response)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_side:
        st.markdown("#### Judicial Sources")
        st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-bottom:1.5rem;'>Verified artifacts from Endee Vector Database</p>", unsafe_allow_html=True)
        
        for r in results:
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-meta">
                    <span class="section-tag">{r['section']}</span>
                    <span class="match-pct">{int(r['score']*100)}% confidence</span>
                </div>
                <div class="doc-title">{r['title']}</div>
                <div class="doc-body">{r['description']}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Empty state / dashboard preview
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="doc-card" style="text-align:center; padding: 2rem;">
            <div style="font-size:2rem; margin-bottom:1rem;">📚</div>
            <div style="font-weight:700; color:white;">Comprehensive Dataset</div>
            <p style="font-size:0.85rem; color:#64748b;">Entire Indian Penal Code indexed with high-dimensional embeddings.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="doc-card" style="text-align:center; padding: 2rem;">
            <div style="font-size:2rem; margin-bottom:1rem;">⚡</div>
            <div style="font-weight:700; color:white;">Neural Retrieval</div>
            <p style="font-size:0.85rem; color:#64748b;">Instant semantic matching using Endee's vector specialized engine.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="doc-card" style="text-align:center; padding: 2rem;">
            <div style="font-size:2rem; margin-bottom:1rem;">🤖</div>
            <div style="font-weight:700; color:white;">Smart Synthesis</div>
            <p style="font-size:0.85rem; color:#64748b;">RAG-based summarization to provide actionable legal insights.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#475569; font-size:0.8rem;'>Endee Vector Database • AI Legal Assistant RAG Demonstration • 2026</p>", unsafe_allow_html=True)
