"""
NUST AI Assistant — Auto-loads documents from /documents folder on startup.
Users see only the chat interface.
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NUST AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #05060f; color: #e2e2f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0b18 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }

.sidebar-logo {
    text-align: center;
    padding: 1.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon { font-size: 3rem; display: block; margin-bottom: 0.6rem; }
.sidebar-logo-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem; font-weight: 700; color: #fff;
}
.sidebar-logo-sub {
    font-size: 0.7rem; color: #3a3a5a;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.3rem;
}

.sidebar-section {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #3a3a5a; margin-bottom: 0.6rem; margin-top: 1.2rem;
}

.status-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 0.55rem 0.9rem;
    font-size: 0.82rem; color: #8888aa; margin-bottom: 0.5rem;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dot-yellow { background: #eab308; box-shadow: 0 0 6px #eab308; }
.dot-red { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

/* Chat header */
.chat-header {
    background: #08091a;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding: 1rem 2rem;
}

/* Message bubbles */
[data-testid="stChatMessage"] { background: transparent !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #0e0f1f !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #e2e2f0 !important;
    border-radius: 12px !important;
}

/* Sidebar text input */
input[type="password"], input[type="text"] {
    background: #0e0f1f !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #e2e2f0 !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from utils.pdf_parser import load_all_documents
from utils.vector_store import (
    create_vector_store, search_vector_store,
    save_vector_store, load_vector_store, vector_store_exists
)
from utils.rag_chain import generate_answer

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "kb_doc_count" not in st.session_state:
    st.session_state.kb_doc_count = 0

# ── Auto-load knowledge base on startup ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """
    Load knowledge base once on startup.
    Priority: saved vector_store/ → build from documents/ folder
    """
    # Try loading pre-built vector store first (faster)
    if vector_store_exists():
        index, chunks = load_vector_store()
        return index, chunks, -1  # -1 = loaded from cache

    # Build from documents folder
    all_chunks, doc_names = load_all_documents("documents")
    if all_chunks:
        index, chunks = create_vector_store(all_chunks)
        save_vector_store(index, chunks)  # Save for next time
        return index, chunks, len(doc_names)

    return None, [], 0


# Load on startup
with st.spinner("🎓 Loading NUST knowledge base..."):
    index, chunks, doc_count = load_knowledge_base()

if index is not None:
    st.session_state.vector_index = index
    st.session_state.chunks = chunks
    st.session_state.kb_ready = True
    st.session_state.kb_doc_count = doc_count

# ── API Keys ──────────────────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY", "")
tavily_key = os.getenv("TAVILY_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">🎓</span>
        <div class="sidebar-logo-title">NUST AI Assistant</div>
        <div class="sidebar-logo-sub">Powered by RAG · Groq · Web Search</div>
    </div>
    """, unsafe_allow_html=True)

    # System status
    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)

    groq_dot = "dot-green" if groq_key else "dot-red"
    groq_label = "Groq LLM — Connected" if groq_key else "Groq LLM — Missing Key"

    tavily_dot = "dot-green" if tavily_key else "dot-yellow"
    tavily_label = "Web Search — Active" if tavily_key else "Web Search — Disabled"

    if st.session_state.kb_ready:
        kb_dot = "dot-green"
        kb_label = f"Knowledge Base — Ready"
    else:
        kb_dot = "dot-red"
        kb_label = "Knowledge Base — No Documents"

    for dot, label in [(groq_dot, groq_label), (tavily_dot, tavily_label), (kb_dot, kb_label)]:
        st.markdown(f"""
        <div class="status-pill">
            <div class="status-dot {dot}"></div>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)

    # API key inputs if missing
    if not groq_key or not tavily_key:
        st.markdown('<div class="sidebar-section">API Keys</div>', unsafe_allow_html=True)

        if not groq_key:
            key_input = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Get free key from console.groq.com"
            )
            if key_input:
                os.environ["GROQ_API_KEY"] = key_input
                groq_key = key_input
                st.rerun()

        if not tavily_key:
            tav_input = st.text_input(
                "Tavily API Key (optional)",
                type="password",
                placeholder="tvly-...",
                help="Get free key from tavily.com"
            )
            if tav_input:
                os.environ["TAVILY_API_KEY"] = tav_input
                st.rerun()

    # Quick questions
    st.markdown('<div class="sidebar-section">Quick Questions</div>', unsafe_allow_html=True)

    quick_questions = [
        "What is the NET test syllabus?",
        "What are NUST admission requirements?",
        "What is the semester fee at NUST?",
        "What scholarships does NUST offer?",
        "How to apply for NUST hostel?",
        "What programs does NUST offer?",
        "What are NUST campus locations?",
        "How is grading done at NUST?",
    ]

    for q in quick_questions:
        if st.button(q, key=f"qq_{q}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    # Clear chat
    st.markdown("---")
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="font-size:0.7rem;color:#2a2a4a;text-align:center;margin-top:1rem;line-height:1.6;">
    Answers from NUST documents<br>+ web search fallback<br>
    <span style="color:#3a3a5a;">nust.edu.pk</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="chat-header" style="display:flex;align-items:center;gap:1rem;">
    <div style="font-size:1.8rem;">🎓</div>
    <div>
        <div style="font-family:'Outfit',sans-serif;font-weight:700;
        font-size:1.05rem;color:#fff;">NUST AI Assistant</div>
        <div style="font-size:0.75rem;color:#3a3a5a;">
        Ask anything about admissions, fees, scholarships, hostel, academics & more
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# No API key warning
if not groq_key:
    st.warning("⚠️ Enter your **Groq API Key** in the sidebar to start chatting. Get it free at **console.groq.com**")

# No knowledge base warning
if not st.session_state.kb_ready:
    st.info("📂 No documents found in the `documents/` folder. Add NUST PDF files and restart the app.")

# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem 1.5rem;">
        <div style="font-size:3.5rem;margin-bottom:0.8rem;">👋</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.5rem;
        font-weight:700;color:#fff;margin-bottom:0.5rem;">
            How can I help you today?
        </div>
        <div style="font-size:0.9rem;color:#4a4a6a;max-width:400px;
        margin:0 auto;line-height:1.6;">
            I have complete knowledge about NUST Pakistan.
            Ask me anything about admissions, programs, fees, or campus life.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📝", "Admissions", "What are the admission requirements and NET test details?"),
        ("💰", "Fee Structure", "What is the semester fee structure for engineering?"),
        ("🏆", "Scholarships", "What scholarships and financial aid does NUST offer?"),
        ("🏠", "Hostel", "How do I apply for hostel accommodation at NUST?"),
    ]
    for col, (icon, title, question) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div style="background:#0e0f1f;border:1px solid rgba(255,255,255,0.07);
            border-radius:14px;padding:1.2rem 1rem;text-align:center;margin-bottom:0.5rem;
            cursor:pointer;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700;font-size:0.85rem;color:#e2e2f0;
                margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.75rem;color:#4a4a6a;line-height:1.4;">{question[:40]}...</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Ask about {title}", key=f"card_{title}", use_container_width=True):
                st.session_state["prefill"] = question
                st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Display messages ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🎓"):
            src = msg.get("source", "")
            if src == "document":
                st.markdown(
                    '<span style="background:rgba(34,197,94,0.1);color:#4ade80;'
                    'border:1px solid rgba(34,197,94,0.2);padding:0.15rem 0.6rem;'
                    'border-radius:4px;font-size:0.68rem;font-weight:700;'
                    'letter-spacing:0.08em;text-transform:uppercase;">'
                    '📄 From Documents</span>',
                    unsafe_allow_html=True
                )
            elif src == "web":
                st.markdown(
                    '<span style="background:rgba(59,130,246,0.1);color:#60a5fa;'
                    'border:1px solid rgba(59,130,246,0.2);padding:0.15rem 0.6rem;'
                    'border-radius:4px;font-size:0.68rem;font-weight:700;'
                    'letter-spacing:0.08em;text-transform:uppercase;">'
                    '🌐 From Web Search</span>',
                    unsafe_allow_html=True
                )
            st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask anything about NUST...", disabled=not groq_key)

if prefill:
    user_input = prefill

# ── Generate response ─────────────────────────────────────────────────────────
if user_input and groq_key:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Search documents
                context_chunks = []
                if st.session_state.vector_index is not None:
                    context_chunks = search_vector_store(
                        user_input,
                        st.session_state.vector_index,
                        st.session_state.chunks,
                        top_k=6,
                    )

                # Chat history
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                # Generate answer
                answer, source = generate_answer(user_input, context_chunks, history)

                # Show source badge
                if source == "document":
                    st.markdown(
                        '<span style="background:rgba(34,197,94,0.1);color:#4ade80;'
                        'border:1px solid rgba(34,197,94,0.2);padding:0.15rem 0.6rem;'
                        'border-radius:4px;font-size:0.68rem;font-weight:700;'
                        'letter-spacing:0.08em;text-transform:uppercase;">'
                        '📄 From Documents</span>',
                        unsafe_allow_html=True
                    )
                elif source == "web":
                    st.markdown(
                        '<span style="background:rgba(59,130,246,0.1);color:#60a5fa;'
                        'border:1px solid rgba(59,130,246,0.2);padding:0.15rem 0.6rem;'
                        'border-radius:4px;font-size:0.68rem;font-weight:700;'
                        'letter-spacing:0.08em;text-transform:uppercase;">'
                        '🌐 From Web Search</span>',
                        unsafe_allow_html=True
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source": source,
                })

            except Exception as e:
                err = str(e)
                if "GROQ_API_KEY" in err or "api_key" in err.lower():
                    st.error("❌ Invalid Groq API key. Please re-enter in the sidebar.")
                else:
                    st.error(f"❌ Error: {err}")
