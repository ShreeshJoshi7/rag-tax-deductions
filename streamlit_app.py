"""
Interactive Streamlit front-end for the ATO tax-deduction RAG assistant.

Drop this file into the same folder as your existing ingest.py / query.py
(it expects the Chroma DB to already exist at db/chroma_db -- run your
ingest.py first if you haven't).

Run with:
    streamlit run streamlit_app.py

Requires (on top of what query.py already needs):
    pip install streamlit python-dotenv
"""

import os
import json
import time
import uuid
from datetime import datetime

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config -- keep these in sync with your ingest.py / query.py
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "db/chroma_db"
GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TOP_K = 3
DEFAULT_MAX_DISTANCE = 0.50
MAX_HISTORY_TURNS = 4
CONVERSATIONS_FILE = "conversations.json"

SUGGESTED_QUESTIONS = [
    "What car expenses can I claim?",
    "Cents per km vs logbook method?",
    "Can I claim working from home costs?",
    "What education expenses are deductible?",
]

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Australian \
individual income tax deductions, using ONLY the context provided below. The context \
comes from official Australian Taxation Office (ATO) guidance.

Rules you must follow:
1. Answer using only the information in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say so \
clearly instead of guessing -- respond with something like "I don't have enough \
information in my knowledge base to answer that confidently."
3. Where relevant, mention specific conditions, thresholds, or exceptions from the \
context rather than giving an overly general answer.
4. Keep the answer clear and in plain English -- the user may not know tax terminology.
5. Do not give personal financial or legal advice framed as certain; present it as \
general information based on the ATO guidance provided.
6. If earlier turns of the conversation are provided, use them only to understand \
what the user is referring to (e.g. "it" or "that"). Still answer only from the \
context given for THIS turn -- do not pull facts from earlier turns' context.
7. Answer directly and concisely. Do not show step-by-step reasoning.
8. Aim for a focused answer of around 120-180 words. Use short paragraphs or a brief \
bullet list -- not one long essay, but not a single sentence either.
9. This assistant's scope is specifically work-related employee deductions (car and \
transport, tools/equipment, clothing, working from home, education/training, and \
similar). If the context mentions a deduction category outside this scope (e.g. gifts \
and donations, investment/super/insurance deductions, cost of managing tax affairs), \
do not elaborate on it. Briefly note that it is outside what this assistant covers in \
depth, and focus on the work-related categories.
10. Do not use citation markup like 【1†L1-L4】. If you reference a source, just say \
"(Source 1)" in plain text."""


NO_RELEVANT_CONTEXT_MESSAGE = (
    "I don't have enough relevant information in my knowledge base to answer that "
    "confidently. This assistant currently only covers work-related tax deductions "
    "(car expenses, tools/equipment, clothing, working from home, education, and "
    "similar). Your question may be outside that scope, or about a different area "
    "of tax law."
)

REWRITE_PROMPT = """Given the conversation so far and a new follow-up question, \
rewrite the follow-up as a standalone question that makes sense without needing the \
earlier turns. Keep it short and preserve the user's intent. If the follow-up is \
already a standalone question, just return it unchanged.

Conversation so far:
{history}

Follow-up question: {question}

Standalone question:"""

# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATO Deduction Assistant",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 20% -10%, #171a24 0%, #0d0f15 55%); }
    section[data-testid="stSidebar"] {
        background: #11141c; border-right: 1px solid #242938;
    }
    .main-title { font-size: 2rem; font-weight: 800; margin-bottom: 0; letter-spacing: -0.02em; }
    .subtitle { color: #8b93a7; font-size: 0.95rem; margin-top: 2px; margin-bottom: 1.4rem; }

    .stat-card {
        background: #161a24; border: 1px solid #242938; border-radius: 12px;
        padding: 10px 14px; text-align: center;
    }
    .stat-num { font-size: 1.3rem; font-weight: 700; color: #7dd3fc; }
    .stat-label { font-size: 0.72rem; color: #8b93a7; text-transform: uppercase; letter-spacing: .04em; }

    .source-chip {
        display: inline-block; background: #1c2130; color: #7dd3fc;
        border-radius: 999px; padding: 2px 10px; font-size: 0.75rem;
        margin: 2px 4px 2px 0; border: 1px solid #2a3142;
    }
    .below-threshold { color: #f59e0b; font-size: 0.75rem; }
    .badge { border-radius: 999px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; display: inline-block; }
    .badge-strong { background: #113a26; color: #4ade80; border: 1px solid #1d5c3c; }
    .badge-partial { background: #3a2d11; color: #fbbf24; border: 1px solid #5c4a1d; }
    .meta-line { color: #6b7284; font-size: 0.75rem; margin-top: 6px; }

    div[data-testid="stChatMessage"] { border-radius: 14px; }

    .conv-row button {
        text-align: left !important; justify-content: flex-start !important;
    }
    .sidebar-section-label {
        color: #6b7284; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: .05em; margin: 14px 0 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Conversation persistence (simple JSON file -- fine for a solo/small-group project)
# ---------------------------------------------------------------------------
def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_conversations(conversations):
    try:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
    except OSError:
        pass  # non-fatal -- app still works, just won't persist this save


def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New chat",
        "created": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
        "rag_history": [],
    }
    st.session_state.active_id = conv_id
    save_conversations(st.session_state.conversations)


def delete_conversation(conv_id):
    st.session_state.conversations.pop(conv_id, None)
    save_conversations(st.session_state.conversations)
    if st.session_state.active_id == conv_id:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.active_id = remaining[0] if remaining else None


def title_from_question(question, max_len=32):
    q = question.strip().rstrip("?")
    return (q[:max_len] + "…") if len(q) > max_len else q


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()
if "active_id" not in st.session_state:
    ids = list(st.session_state.conversations.keys())
    st.session_state.active_id = ids[0] if ids else None
if st.session_state.active_id is None:
    new_conversation()

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_llm():
    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is not set. Add it to a `.env` file in this folder "
            "(GROQ_API_KEY=your-key) or set it in your terminal before launching "
            "Streamlit, then restart the app."
        )
        st.stop()
    return ChatGroq(model=GROQ_MODEL, temperature=0.1, max_tokens=700)


# ---------------------------------------------------------------------------
# Pipeline logic (mirrors query.py)
# ---------------------------------------------------------------------------
def format_history(history):
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def rewrite_followup(llm, question, history):
    if not history:
        return question
    prompt = REWRITE_PROMPT.format(history=format_history(history), question=question)
    response = llm.invoke([("human", prompt)])
    rewritten = response.content.strip().strip('"')
    return rewritten if rewritten else question


def retrieve_context(vectorstore, question, k):
    return vectorstore.similarity_search_with_score(question, k=k)


def is_relevant(results, max_distance):
    if not results:
        return False
    return min(score for _, score in results) <= max_distance


def build_prompt(question, results, history):
    context_blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[Source {i}: {source}]\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    history_block = ""
    if history:
        history_block = f"Earlier in this conversation:\n{format_history(history)}\n\n"

    return f"""{history_block}Context for this question:
{context_text}

Question: {question}

Answer the question using only the context above."""


def render_sources(source_rows, show_scores):
    with st.expander("📎 Sources checked", expanded=False):
        for i, (name, score, used) in enumerate(source_rows, start=1):
            flag = "" if used else ' <span class="below-threshold">below threshold</span>'
            score_str = f" · {score:.3f}" if show_scores else ""
            st.markdown(
                f'<span class="source-chip">[{i}] {name}{score_str}</span>{flag}',
                unsafe_allow_html=True,
            )


def confidence_badge(source_rows):
    if not source_rows:
        return ""
    best = min(score for _, score, _ in source_rows)
    if best <= 0.25:
        return '<span class="badge badge-strong">● Strong match</span>'
    return '<span class="badge badge-partial">● Partial match</span>'


def export_conversation_md(conv):
    lines = [f"# {conv['title']}", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for m in conv["messages"]:
        who = "**You**" if m["role"] == "user" else "**Assistant**"
        lines.append(f"{who}: {m['content']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar -- conversation list + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧾 ATO Assistant")

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    search = st.text_input("Search chats", placeholder="🔍 Search...", label_visibility="collapsed")

    st.markdown('<div class="sidebar-section-label">Conversations</div>', unsafe_allow_html=True)

    convs_sorted = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1]["created"],
        reverse=True,
    )
    if search:
        convs_sorted = [(cid, c) for cid, c in convs_sorted if search.lower() in c["title"].lower()]

    if not convs_sorted:
        st.caption("No conversations yet.")

    for conv_id, conv in convs_sorted:
        is_active = conv_id == st.session_state.active_id
        row = st.columns([5, 1])
        with row[0]:
            label = ("📌 " if is_active else "") + conv["title"]
            if st.button(label, key=f"open_{conv_id}", use_container_width=True):
                st.session_state.active_id = conv_id
                st.rerun()
        with row[1]:
            if st.button("🗑", key=f"del_{conv_id}", help="Delete this chat"):
                delete_conversation(conv_id)
                st.rerun()

    st.markdown('<div class="sidebar-section-label">Settings</div>', unsafe_allow_html=True)
    with st.expander("⚙️ Retrieval tuning", expanded=False):
        top_k = st.slider("Chunks retrieved", 1, 6, DEFAULT_TOP_K)
        max_distance = st.slider("Relevance threshold", 0.10, 0.80, DEFAULT_MAX_DISTANCE, 0.01)
        show_scores = st.toggle("Show retrieval scores", value=True)

    st.markdown('<div class="sidebar-section-label">Knowledge base</div>', unsafe_allow_html=True)
    if os.path.exists(PERSIST_DIR):
        st.success("Vector store ready", icon="✅")
    else:
        st.error("Missing -- run ingest.py first", icon="⚠️")

    active_conv = st.session_state.conversations.get(st.session_state.active_id)
    if active_conv and active_conv["messages"]:
        st.download_button(
            "⬇️ Export this chat (.md)",
            data=export_conversation_md(active_conv),
            file_name=f"{active_conv['title'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(f"Model: `{GROQ_MODEL}` via Groq")
    st.caption("Scope: car, WFH, tools, clothing, education expenses")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">Ask about your tax deductions</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Grounded in official ATO guidance · answers cite their sources.</p>',
    unsafe_allow_html=True,
)

vectorstore = get_vectorstore()
llm = get_llm()

active_conv = st.session_state.conversations[st.session_state.active_id]

total_convs = len(st.session_state.conversations)
total_questions = sum(
    len([m for m in c["messages"] if m["role"] == "user"])
    for c in st.session_state.conversations.values()
)
stat_cols = st.columns(3)
for col, (num, label) in zip(
    stat_cols,
    [
        (total_convs, "Conversations"),
        (total_questions, "Questions asked"),
        (vectorstore._collection.count(), "Chunks in KB"),
    ],
):
    with col:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{num}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# Render existing conversation
for msg in active_conv["messages"]:
    with st.chat_message(msg["role"], avatar="🧾" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta_bits = []
            if msg.get("sources"):
                meta_bits.append(confidence_badge(msg["sources"]))
            if msg.get("elapsed"):
                meta_bits.append(f'<span class="meta-line">⚡ {msg["elapsed"]:.1f}s</span>')
            if meta_bits:
                st.markdown(" &nbsp; ".join(meta_bits), unsafe_allow_html=True)
            if msg.get("sources"):
                render_sources(msg["sources"], show_scores)

# Suggested questions for a fresh chat
if not active_conv["messages"]:
    st.caption("Try one of these:")
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, sq in zip(chip_cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(sq, use_container_width=True, key=f"suggest_{sq}"):
                st.session_state.pending_question = sq
                st.rerun()

# ---------------------------------------------------------------------------
# Handle a question (from chat input or a suggested-question chip)
# ---------------------------------------------------------------------------
def handle_question(question):
    active_conv["messages"].append({"role": "user", "content": question})
    if active_conv["title"] == "New chat":
        active_conv["title"] = title_from_question(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧾"):
        start = time.time()
        with st.spinner("Retrieving relevant ATO guidance..."):
            retrieval_question = (
                rewrite_followup(llm, question, active_conv["rag_history"])
                if active_conv["rag_history"]
                else question
            )
            results = retrieve_context(vectorstore, retrieval_question, top_k)
            relevant = is_relevant(results, max_distance)

        source_rows = [
            [os.path.basename(doc.metadata.get("source", "unknown")), score, score <= max_distance]
            for doc, score in results
        ]

        if not relevant:
            answer = NO_RELEVANT_CONTEXT_MESSAGE
            st.markdown(answer)
        else:
            user_prompt = build_prompt(question, results, active_conv["rag_history"])
            messages = [("system", SYSTEM_PROMPT), ("human", user_prompt)]
            answer = st.write_stream(chunk.content for chunk in llm.stream(messages))

        elapsed = time.time() - start

        meta_bits = []
        if source_rows:
            meta_bits.append(confidence_badge(source_rows))
        meta_bits.append(f'<span class="meta-line">⚡ {elapsed:.1f}s</span>')
        st.markdown(" &nbsp; ".join(meta_bits), unsafe_allow_html=True)
        if source_rows:
            render_sources(source_rows, show_scores)

    active_conv["messages"].append(
        {"role": "assistant", "content": answer, "sources": source_rows, "elapsed": elapsed}
    )
    if relevant:
        active_conv["rag_history"].append({"question": question, "answer": answer})
        active_conv["rag_history"] = active_conv["rag_history"][-MAX_HISTORY_TURNS:]

    save_conversations(st.session_state.conversations)


if not os.path.exists(PERSIST_DIR):
    st.info("Add your knowledge base first by running `ingest.py`, then reload this page.")
else:
    pending = st.session_state.pop("pending_question", None)
    if pending:
        handle_question(pending)
        st.rerun()

    if question := st.chat_input("Ask about a work-related deduction..."):
        handle_question(question)
        st.rerun()