"""
Streamlit Frontend for Healthcare AI Assistant.

A premium chat UI that connects to the FastAPI backend at localhost:8000.
Features: document ingestion, multi-turn chat, source citations,
confidence badges, appointment tool indicators, and health status sidebar.
"""

import time
import streamlit as st
import requests
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

CONFIDENCE_COLORS = {
    "high":   "#10b981",  # emerald
    "medium": "#f59e0b",  # amber
    "low":    "#ef4444",  # red
    "none":   "#6b7280",  # gray
}

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark healthcare theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.2);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.15), rgba(129,140,248,0.15));
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px 0;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Chat messages ── */
.chat-bubble-user {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    color: #e0f2fe;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(29, 78, 216, 0.3);
}
.chat-bubble-assistant {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99, 179, 237, 0.25);
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin: 8px 0;
    max-width: 90%;
    color: #e2e8f0;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.chat-timestamp {
    color: #475569;
    font-size: 0.72rem;
    margin-top: 6px;
}

/* ── Confidence badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 6px;
}

/* ── Source cards ── */
.source-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
}
.source-doc-name {
    color: #63b3ed;
    font-weight: 600;
    font-size: 0.82rem;
}
.source-chunk {
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.5;
}

/* ── Tool response card ── */
.tool-card {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(16, 185, 129, 0.05));
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 0.85rem;
    color: #6ee7b7;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-healthy {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
}
.status-error {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #f87171;
}

/* ── Input area ── */
[data-testid="stTextInput"] input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 179, 237, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(99, 179, 237, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(99, 179, 237, 0.15) !important;
    margin: 16px 0 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 10px;
    padding: 12px;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }

/* ── Scrollable chat container ── */
.chat-container {
    max-height: 58vh;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: rgba(99, 179, 237, 0.3) transparent;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "health_info" not in st.session_state:
    st.session_state.health_info = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def call_health() -> dict | None:
    """GET /health and return the JSON or None on error."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_ingest(reset: bool = False) -> dict | None:
    """POST /ingest and return the JSON or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE}/ingest",
            json={"reset": reset},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        try:
            return e.response.json()
        except Exception:
            return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def call_ask(question: str) -> dict | None:
    """POST /ask and return the JSON or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        try:
            data = e.response.json()
            # FastAPI wraps errors in 'detail'
            detail = data.get("detail", data)
            if isinstance(detail, dict):
                return {"error": detail.get("message", str(detail))}
            return {"error": str(detail)}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def confidence_badge(level: str) -> str:
    color = CONFIDENCE_COLORS.get(level, "#6b7280")
    return (
        f'<span class="badge" style="background:{color}22;'
        f'border:1px solid {color};color:{color};">'
        f'⬤ {level.upper()}</span>'
    )


def render_message(msg: dict):
    """Render a single chat message (user or assistant)."""
    role = msg["role"]
    ts = msg.get("timestamp", "")

    if role == "user":
        st.markdown(
            f'<div class="chat-bubble-user">'
            f'<strong>You</strong><br>{msg["content"]}'
            f'<div class="chat-timestamp">{ts}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        data = msg.get("data", {})
        answer = data.get("answer", msg["content"])
        sources = data.get("sources", [])
        confidence = data.get("confidence", "low")
        tool_used = data.get("tool_used")
        tool_response = data.get("tool_response")
        model_used = data.get("model_used", "")
        error = data.get("error")

        if error:
            st.markdown(
                f'<div class="chat-bubble-assistant">'
                f'⚠️ <strong>Error:</strong> {error}'
                f'<div class="chat-timestamp">{ts}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

        # Main answer bubble
        tool_tag = ""
        if tool_used:
            tool_tag = (
                '<span class="badge" style="background:rgba(52,211,153,0.15);'
                'border:1px solid #34d399;color:#34d399;">'
                f'🔧 {tool_used}</span>'
            )

        st.markdown(
            f'<div class="chat-bubble-assistant">'
            f'{confidence_badge(confidence)}{tool_tag}'
            f'<div style="margin-top:10px;line-height:1.7;">{answer}</div>'
            f'<div class="chat-timestamp" style="margin-top:8px;">'
            f'🤖 {model_used} · {ts}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Appointment tool response
        if tool_response:
            slots = ", ".join(tool_response.get("available_slots", []))
            dept = tool_response.get("department", "")
            date = tool_response.get("date", "")
            instr = tool_response.get("booking_instructions", "")
            st.markdown(
                f'<div class="tool-card">'
                f'📅 <strong>Mock Scheduling Tool Response</strong><br>'
                f'<strong>Department:</strong> {dept} · <strong>Date:</strong> {date}<br>'
                f'<strong>Available Slots:</strong> {slots}<br>'
                f'<strong>Booking:</strong> {instr}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Source citations
        if sources:
            with st.expander(f"📄 {len(sources)} source(s) used", expanded=False):
                for src in sources:
                    doc = src.get("document", "unknown")
                    chunk = src.get("chunk", "")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-doc-name">📎 {doc}</div>'
                        f'<div class="source-chunk">{chunk}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Healthcare AI")
    st.markdown("---")

    # ── Health status ──
    st.markdown("### 🔍 System Status")
    if st.button("Check Health", key="btn_health", use_container_width=True):
        with st.spinner("Pinging API..."):
            st.session_state.health_info = call_health()

    if st.session_state.health_info:
        h = st.session_state.health_info
        if h.get("status") == "healthy":
            st.markdown(
                '<span class="status-pill status-healthy">● API Online</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-error">● API Error</span>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**Model:** `{h.get('model','—')}`")
        st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2`")
        st.markdown(f"**Vector DB:** `ChromaDB`")
    else:
        st.markdown(
            '<span class="status-pill status-error">● Not checked</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Ingestion ──
    st.markdown("### 📥 Knowledge Base")
    reset_flag = st.checkbox("Reset before ingesting", value=False, key="reset_chk")

    if st.button("Ingest Documents", key="btn_ingest", use_container_width=True):
        with st.spinner("Ingesting documents… this may take a minute."):
            result = call_ingest(reset=reset_flag)

        if result and result.get("status") == "success":
            st.session_state.ingested = True
            st.success(
                f"✅ Ingested {result['documents_loaded']} docs → "
                f"{result['chunks_created']} chunks"
            )
        else:
            msg = result.get("message", "Unknown error") if result else "API unreachable"
            st.error(f"❌ {msg}")

    if st.session_state.ingested:
        st.markdown(
            '<span class="status-pill status-healthy">● Knowledge Base Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("⚠️ Ingest documents before asking questions.")

    st.markdown("---")

    # ── Conversation controls ──
    st.markdown("### 💬 Conversation")
    if st.button("Clear Chat", key="btn_clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "What is the weight limit for lifting after discharge?",
        "How early should I arrive before my appointment?",
        "Can I refill an opioid prescription via the portal?",
        "Book a cardiology appointment for Monday",
        "How long are adult medical records kept?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sq_{hash(q)}", use_container_width=True):
            st.session_state["prefill_question"] = q
            st.rerun()

    st.markdown("---")
    st.caption("Built with FastAPI · LangChain · ChromaDB · Groq LLM")


def submit_chat():
    """Callback to handle chat submission (from button or Enter key)."""
    q = st.session_state.user_input_field.strip()
    if q:
        st.session_state.pending_question = q
    # Clear the input field in session state
    st.session_state.user_input_field = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main area — header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1 class="hero-title">🏥 Healthcare AI Assistant</h1>
  <p class="hero-subtitle">
    RAG-powered · LangChain · ChromaDB · Groq llama-3.1-8b-instant ·
    Appointment routing · Source citations
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────────────────────────────────────
chat_placeholder = st.container()

with chat_placeholder:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:48px 0;color:#475569;">
            <div style="font-size:3rem;margin-bottom:16px;">💬</div>
            <div style="font-size:1.1rem;font-weight:500;color:#64748b;">
                Start by ingesting documents, then ask a healthcare question.
            </div>
            <div style="margin-top:10px;font-size:0.85rem;color:#475569;">
                Try the sample questions in the sidebar →
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Input area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Handle pre-filled questions from sample buttons
prefill = st.session_state.pop("prefill_question", "")

col_input, col_send = st.columns([5, 1])
with col_input:
    # We remove `value=prefill` so it doesn't conflict with session state clearing.
    # Instead, we push prefill directly into the processing pipeline if present.
    user_input = st.text_input(
        "Ask a healthcare question…",
        placeholder="e.g. Can I request a medication refill through telehealth?",
        label_visibility="collapsed",
        key="user_input_field",
        on_change=submit_chat,
    )
with col_send:
    send_clicked = st.button("Send ➤", key="btn_send", use_container_width=True, on_click=submit_chat)

# ─────────────────────────────────────────────────────────────────────────────
# Process submission
# ─────────────────────────────────────────────────────────────────────────────
# question comes either from a sidebar prefill button, or from the callback
question = prefill or st.session_state.pop("pending_question", "")

if question:
    ts_now = datetime.now().strftime("%H:%M")

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": ts_now,
    })

    # Call the API
    with st.spinner("🔍 Searching knowledge base…"):
        start = time.time()
        result = call_ask(question)
        elapsed = round(time.time() - start, 2)

    # Build assistant message
    if result is None:
        data = {"error": "Could not reach the API. Make sure the FastAPI server is running on port 8000."}
    elif "error" in result:
        data = result
    else:
        data = result

    st.session_state.messages.append({
        "role": "assistant",
        "content": data.get("answer", data.get("error", "")),
        "data": data,
        "timestamp": f"{ts_now} ({elapsed}s)",
    })

    st.rerun()
