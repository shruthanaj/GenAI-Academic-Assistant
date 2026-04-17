import sys
import os
import time

# Suppress PyTorch/TensorFlow initialization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from backend.router_agent import RouterAgent
from rag.rag_pipeline import RAGPipeline
from utils.pdf_loader import load_file_text

st.set_page_config(
    page_title="GenAI Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    /* ── Global font & base ── */
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    /* ── App backgrounds ── */
    .stApp                          { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%) !important; }
    .main .block-container          { background: transparent !important; }

    /* ── Sidebar full dark ── */
    [data-testid="stSidebar"]                          { background: #0d1117 !important; border-right: 1px solid #1e2a3a; }
    [data-testid="stSidebar"] > div                    { background: #0d1117 !important; }
    [data-testid="stSidebarContent"]                   { background: #0d1117 !important; }

    /* ── ALL text to light ── */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span  { color: #cbd5e1 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"]                     { background: #131928 !important; border: 1.5px dashed #2d3f5a !important; border-radius: 10px !important; }
    [data-testid="stFileUploader"] section             { background: #131928 !important; }
    [data-testid="stFileUploader"] *                   { color: #94a3b8 !important; }
    [data-testid="stFileUploaderDropzone"]             { background: #131928 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { color: #64748b !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #1e2a3a !important; color: #60a5fa !important;
        border: 1px solid #2d4a6a !important; border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #1e3a5f !important; border-color: #60a5fa !important;
        box-shadow: 0 0 12px rgba(96,165,250,0.25) !important;
    }

    /* ── Input/textarea/selectbox/radio ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stRadio > div                  { background: #131928 !important; color: #e2e8f0 !important; border-color: #1e2a3a !important; }
    .stSelectbox > div > div > div  { background: #131928 !important; color: #e2e8f0 !important; }
    div[data-baseweb="select"]      { background: #131928 !important; }
    div[data-baseweb="select"] *    { background: #131928 !important; color: #e2e8f0 !important; }
    ul[data-baseweb="menu"]         { background: #1a1f2e !important; border: 1px solid #2d3f5a !important; }
    li[role="option"]               { background: #1a1f2e !important; color: #e2e8f0 !important; }
    li[role="option"]:hover         { background: #1e3a5f !important; }

    /* ── Radio buttons ── */
    .stRadio label                  { color: #cbd5e1 !important; }
    [data-baseweb="radio"] div      { border-color: #60a5fa !important; }

    /* ── Form ── */
    [data-testid="stForm"]          { background: #0d1424 !important; border: 1px solid #1e2a3a !important; border-radius: 12px !important; padding: 16px !important; }
    .stForm                         { background: #0d1424 !important; }
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1e3a5f, #2d4a7a) !important;
        color: #60a5fa !important; border: 1px solid #3a6aaa !important;
        border-radius: 8px !important; font-weight: 600 !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 0 16px rgba(96,165,250,0.35) !important;
        background: linear-gradient(135deg, #2d4a7a, #3a5a8a) !important;
    }

    /* ── Chat ── */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] > div {
        background: #131928 !important; border: 1px solid #1e2a3a !important;
        color: #e2e8f0 !important; border-radius: 12px !important;
    }
    [data-testid="stChatMessageContent"] { background: transparent !important; }
    [data-testid="stChatMessage"]        { background: #111827 !important; border-radius: 10px !important; border: 1px solid #1e2a3a !important; margin-bottom: 8px; }

    /* ── Divider ── */
    hr { border-color: #1e2a3a !important; }

    /* ── Alerts / Success / Error ── */
    [data-testid="stAlert"]         { background: #0d1424 !important; border-radius: 8px !important; }

    /* ── Spinner ── */
    .stSpinner > div                { border-top-color: #60a5fa !important; }

    /* ── Markdown text in main area ── */
    .stMarkdown, .stMarkdown *      { color: #e2e8f0; }

    /* ── Agent badge chips ── */
    .agent-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 8px;
    }
    .badge-explain { background: #1e3a5f; color: #60a5fa; }
    .badge-exam    { background: #3a1e1e; color: #f87171; }
    .badge-summary { background: #1e3a2e; color: #34d399; }
    .badge-plan    { background: #3a2e1e; color: #fbbf24; }

    /* ── Sidebar title ── */
    .sidebar-title { color: #60a5fa !important; font-weight: 700; font-size: 1.1rem; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar            { width: 6px; }
    ::-webkit-scrollbar-track      { background: #0d1117; }
    ::-webkit-scrollbar-thumb      { background: #2d3f5a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3d5f8a; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
# Study Planner state
if "planner_awaiting_assessment" not in st.session_state:
    st.session_state.planner_awaiting_assessment = False
if "planner_original_question" not in st.session_state:
    st.session_state.planner_original_question = "Create a personalized study plan"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">📂 Upload Study Material</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Study Material",
        type=["pdf", "txt", "docx", "xlsx", "csv", "pptx", "json", "md"],
        accept_multiple_files=True
    )

    if uploaded and st.button("⚡ Build Knowledge Base", use_container_width=True):
        all_text = ""
        try:
            for f in uploaded:
                file_name = f.name
                text = load_file_text(f, file_name)
                all_text += f"\n\n=== {file_name} ===\n\n" + text

            with st.spinner("Indexing your documents…"):
                rag = RAGPipeline()
                rag.build_index(all_text)
                st.session_state.rag = rag

            st.session_state.uploaded_files = sorted({f.name for f in uploaded})
            st.success(f"✅ {len(st.session_state.uploaded_files)} file(s) indexed!")
        except Exception as exc:
            st.error(f"Unable to build knowledge base: {exc}")

    st.divider()
    st.markdown("**🤖 Available Agents**")
    agents_info = [
        ("🔵", "Concept Explainer", "Explains in simple words"),
        ("🔴", "Exam Question", "Generates Q&A"),
        ("🟢", "Notes Summarizer", "Summarizes chapters"),
        ("🟡", "Study Planner", "Makes revision plans"),
    ]
    for icon, name, desc in agents_info:
        st.markdown(f"{icon} **{name}** — {desc}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.planner_awaiting_assessment = False
        st.session_state.planner_original_question = "Create a personalized study plan"
        st.rerun()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎓 GenAI Academic Assistant")
st.caption("Powered by RAG + Agentic Workflows | Ask anything about your study material")

# Chat History
for msg in st.session_state.chat_history:
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant" and "agent" in msg:
            badge_class = {
                "Concept Explainer Agent": "badge-explain",
                "Exam Question Agent": "badge-exam",
                "Notes Summarizer Agent": "badge-summary",
                "Study Planner Agent": "badge-plan",
            }.get(msg["agent"], "badge-explain")
            st.markdown(
                f'<span class="agent-badge {badge_class}">🤖 {msg["agent"]}</span>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

# Input
question = st.chat_input("Ask a question about your subject…")

# ========== HANDLE STUDY PLANNER ASSESSMENT FORM ==========
if st.session_state.planner_awaiting_assessment:
    st.divider()
    st.markdown("### 📋 Student Assessment Form")
    st.markdown("Answer these quick questions so the Study Planner can build a **personalized plan just for you**:")

    with st.form("planner_assessment_form"):
        level = st.radio(
            "1️⃣  What's your current level of understanding?",
            ["Beginner", "Intermediate", "Advanced"],
            horizontal=True,
        )
        topics = st.text_area(
            "2️⃣  Which topics are you strong / weak in? *(optional)*",
            height=90,
            placeholder="e.g., Compiler phases (weak), Symbol Tables (okay), Optimization (needs work)…",
        )
        time_available = st.radio(
            "3️⃣  How much time do you have before your exam?",
            ["< 1 week", "1-2 weeks", "2-3 weeks", "3+ weeks"],
            horizontal=True,
        )
        style = st.selectbox(
            "4️⃣  Preferred learning style?",
            ["Visual (diagrams/charts)",
             "Textual (reading/notes)",
             "Practical (examples/problems)",
             "Mixed/Balanced"],
        )

        submitted = st.form_submit_button("💡 Generate My Personalized Plan", use_container_width=True)

        if submitted:
            from agents.study_planner import StudyPlannerAgent

            # Build assessment string from the form choices
            topics_val = topics.strip() if topics.strip() else "No specific topics mentioned — treat all topics equally"
            assessment_text = f"""• Level: {level}
• Strong/Weak Topics: {topics_val}
• Time Available: {time_available}
• Learning Style: {style}"""

            # Retrieve context using the original question
            original_q = st.session_state.planner_original_question
            ctx = ""
            if st.session_state.rag:
                ctx = st.session_state.rag.retrieve(original_q, limit_tokens=False)

            # Generate the plan RIGHT NOW — no 3rd message needed
            planner = StudyPlannerAgent()
            with st.spinner("🧠 Building your personalized study plan… (this may take ~30s)"):
                plan = planner.create_personalized_plan(original_q, assessment_text, ctx)

            # Push plan into chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": plan,
                "agent": "Study Planner Agent",
            })

            # Reset planner state
            st.session_state.planner_awaiting_assessment = False
            st.session_state.planner_original_question = "Create a personalized study plan"
            st.rerun()

# ========== PROCESS CHAT MESSAGE ==========
# If assessment form is open, block new questions so user must fill it first
if question and st.session_state.planner_awaiting_assessment:
    st.info("⬆️ Please fill out the **Study Planner Assessment Form** above before sending a new message.")
elif question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            context = ""
            if st.session_state.rag:
                context = st.session_state.rag.retrieve(question, limit_tokens=False)

            router = RouterAgent()
            agent_name, answer = router.route_and_answer(question, context)

            # ========== SPECIAL HANDLING FOR STUDY PLANNER ==========
            if agent_name == "Study Planner Agent":
                # Save the original question and show the assessment form
                # Plan is generated directly from the form submit (no 3rd message needed)
                st.session_state.planner_original_question = question
                st.session_state.planner_awaiting_assessment = True
                # answer is already the static intro message from ask_assessment_questions()

            badge_class = {
                "Concept Explainer Agent": "badge-explain",
                "Exam Question Agent": "badge-exam",
                "Notes Summarizer Agent": "badge-summary",
                "Study Planner Agent": "badge-plan",
            }.get(agent_name, "badge-explain")

            st.markdown(
                f'<span class="agent-badge {badge_class}">🤖 {agent_name}</span>',
                unsafe_allow_html=True
            )
            st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "agent": agent_name,
        })
