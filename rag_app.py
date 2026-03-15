import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = ""   # 👈 ADD YOUR KEY HERE
MODEL          = "gpt-4o-mini"
CHUNK_SIZE     = 300
CHUNK_OVERLAP  = 30
TOP_K          = 8

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AskMyPDF", page_icon="📌", layout="centered")

# ─── CSS — PINTEREST VIBES ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

:root {
    --cream:   #fdf6f0;
    --blush:   #f2e0d8;
    --rose:    #e8b4a0;
    --brick:   #c1664a;
    --warm:    #8b3a2a;
    --bark:    #3d2314;
    --sand:    #e8ddd4;
    --white:   #fffcfa;
    --muted:   #9c8880;
    --card:    #ffffff;
}

html, body, .stApp {
    background: var(--cream) !important;
    color: var(--bark);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem 8rem !important;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 4rem 0 2rem;
    animation: riseIn 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes riseIn {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-tag {
    display: inline-block;
    background: var(--blush);
    color: var(--brick);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 99px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 4.2rem;
    font-weight: 700;
    color: var(--bark);
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 1rem;
}
.hero-title em {
    font-style: italic;
    color: var(--brick);
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── PIPELINE STRIP ── */
.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 6px;
    padding: 1rem 1.5rem;
    background: var(--white);
    border: 1px solid var(--sand);
    border-radius: 16px;
    margin: 1.5rem 0;
    box-shadow: 0 2px 12px rgba(61,35,20,0.06);
}
.pipe-step {
    background: var(--blush);
    color: var(--brick);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    padding: 5px 12px;
    border-radius: 99px;
    text-transform: uppercase;
    white-space: nowrap;
}
.pipe-arrow { color: var(--rose); font-size: 0.75rem; }

/* ── UPLOAD CARD ── */
.upload-card {
    background: var(--white);
    border: 1.5px dashed var(--rose);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    margin: 1.5rem 0;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(61,35,20,0.06);
    animation: riseIn 0.8s 0.1s cubic-bezier(0.16,1,0.3,1) both;
}
.upload-card:hover {
    border-color: var(--brick);
    box-shadow: 0 8px 32px rgba(193,102,74,0.12);
}
.upload-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.upload-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: var(--bark);
    margin-bottom: 0.4rem;
}
.upload-hint {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}

/* ── STATS GRID ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1.5rem 0;
    animation: riseIn 0.5s ease both;
}
.stat-card {
    background: var(--white);
    border: 1px solid var(--sand);
    border-radius: 16px;
    padding: 1.1rem 0.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(61,35,20,0.05);
    transition: all 0.25s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(193,102,74,0.12);
    border-color: var(--rose);
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--brick);
    line-height: 1.2;
}
.stat-lbl {
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}

/* ── PROCESSING STEPS ── */
.step {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--blush);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 5px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--warm);
    animation: riseIn 0.3s ease both;
}

/* ── RAG PROCESS BADGE ── */
.rag-badge {
    background: var(--blush);
    border: 1px solid var(--sand);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
}
.rag-tag {
    background: var(--white);
    border: 1px solid var(--rose);
    color: var(--brick);
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: 1px;
    white-space: nowrap;
}
.rewritten-q {
    width: 100%;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 0.85rem;
    color: var(--warm);
    margin-top: 4px;
    padding-left: 4px;
}

/* ── CHAT BUBBLES ── */
.chat-feed {
    display: flex;
    flex-direction: column;
    gap: 18px;
    margin: 1rem 0;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 10px;
    animation: riseIn 0.3s ease both;
}
.bubble-user {
    background: var(--brick);
    color: #fff6f3;
    padding: 14px 20px;
    border-radius: 22px 22px 6px 22px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.65;
    box-shadow: 0 4px 20px rgba(193,102,74,0.25);
    word-wrap: break-word;
    font-family: 'DM Sans', sans-serif;
}
.av-user {
    width: 34px; height: 34px;
    background: var(--brick);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(193,102,74,0.3);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
    animation: riseIn 0.3s ease both;
}
.bubble-bot {
    background: var(--white);
    border: 1px solid var(--sand);
    color: var(--bark);
    padding: 14px 20px;
    border-radius: 22px 22px 22px 6px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.75;
    word-wrap: break-word;
    font-family: 'DM Sans', sans-serif;
    box-shadow: 0 4px 16px rgba(61,35,20,0.07);
}
.av-bot {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--blush), var(--rose));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0; margin-top: 2px;
    box-shadow: 0 2px 8px rgba(193,102,74,0.2);
}

/* ── SOURCE CHUNKS ── */
.chunk {
    background: var(--cream);
    border-left: 3px solid var(--rose);
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.76rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--bark);
    margin-bottom: 0.5rem;
}
.empty-sub { font-size: 0.85rem; color: var(--muted); }

/* ── FIXED INPUT BAR ── */
.input-wrap {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, var(--cream) 65%, transparent);
    padding: 1.5rem;
    z-index: 999;
}
.input-inner {
    max-width: 720px;
    margin: 0 auto;
    background: #fffcfa !important;
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 6px 6px 6px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 8px 32px rgba(61,35,20,0.1);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.input-inner:focus-within {
    border-color: var(--brick);
    box-shadow: 0 8px 32px rgba(193,102,74,0.15);
}

/* ── STREAMLIT OVERRIDES ── */
.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #3d2314 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 10px 0 !important;
    caret-color: var(--brick) !important;
}
.stTextInput > div > div { background: #fffcfa !important; border: none !important; box-shadow: none !important; }
.stTextInput > div > div > input::placeholder { color: #b8a89e !important; }

.stButton > button {
    background: var(--brick) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    box-shadow: 0 4px 14px rgba(193,102,74,0.3) !important;
}
.stButton > button:hover {
    background: var(--warm) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(193,102,74,0.4) !important;
}

/* Clear = ghost */
[data-testid="column"]:last-child .stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--sand) !important;
    color: var(--muted) !important;
    box-shadow: none !important;
}
[data-testid="column"]:last-child .stButton > button:hover {
    border-color: var(--brick) !important;
    color: var(--brick) !important;
    transform: none !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] * { color: var(--muted) !important; }

.stExpander {
    background: var(--white) !important;
    border: 1px solid var(--sand) !important;
    border-radius: 12px !important;
}
details summary {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

.stSpinner > div { border-top-color: var(--brick) !important; }
.stAlert { border-radius: 14px !important; }
div.stMarkdown p { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in {
    "messages": [], "retriever": None, "llm": None,
    "all_texts": [], "chunks_count": 0,
    "pdf_name": None, "pages_count": 0
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the question to improve document retrieval.
Return ONLY the rewritten question.
Question: {question}
""")

rerank_prompt = ChatPromptTemplate.from_template("""
Rate how relevant this chunk is to the question.
Return ONLY a number from 0-10.
Question: {question}
Text: {chunk}
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant analyzing a document.
Use the context below to answer the question as best as you can.
Be detailed and specific in your answer.
Only say "I couldn't find that in the document" if the context has absolutely NO relevant information.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ─── ADVANCED RAG PIPELINE ────────────────────────────────────────────────────
def advanced_rag(question, retriever, all_texts, llm):
    steps = {}

    # Step 1: Query Rewrite
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten_q = rewrite_chain.invoke({"question": question}).strip()
    steps["rewritten"] = rewritten_q

    # Step 2: Hybrid Search (Semantic + BM25)
    semantic_docs = retriever.invoke(rewritten_q)
    text_docs = [Document(page_content=t) for t in all_texts]
    bm25 = BM25Retriever.from_documents(text_docs, k=TOP_K)
    bm25_docs = bm25.invoke(rewritten_q)

    # Also search with original question
    semantic_docs_orig = retriever.invoke(question)
    bm25_docs_orig = bm25.invoke(question)

    # Combine and deduplicate all results
    seen, combined = set(), []
    for doc in semantic_docs + bm25_docs + semantic_docs_orig + bm25_docs_orig:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            combined.append(doc)
    steps["hybrid_count"] = len(combined)
    steps["scores"] = ["hybrid"] * min(TOP_K, len(combined))

    # Use top chunks directly — no re-ranking
    top_docs = combined[:TOP_K]

    # Step 3: Answer
    context = format_docs(top_docs)
    answer = (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question
    })

    return answer, top_docs, steps

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-tag">📌 Advanced RAG · {MODEL}</div>
    <div class="hero-title">Ask<em>My</em>PDF</div>
    <div class="hero-sub">Drop a PDF. Ask anything. Get beautiful answers.</div>
</div>
""", unsafe_allow_html=True)

# ─── PIPELINE STRIP ───────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
    <span class="pipe-step">✍️ Rewrite</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">🔀 Hybrid</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">📊 Re-rank</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">🤖 Answer</span>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ───────────────────────────────────────────────────────────────────
if not st.session_state.retriever:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-icon">📄</div>
        <div class="upload-title">Upload your PDF</div>
        <div class="upload-hint">pdf · up to any size · instant processing</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("✨  Analyse PDF  →", use_container_width=True)

        if process_btn:
            with st.spinner("Processing your PDF..."):
                try:
                    st.markdown('<div class="step">📄 Loading PDF pages...</div>', unsafe_allow_html=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    st.markdown('<div class="step">✂️ Splitting into chunks...</div>', unsafe_allow_html=True)
                    full_text = "\n".join(doc.page_content for doc in docs)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    texts = splitter.split_text(full_text)

                    st.markdown('<div class="step">🧠 Generating embeddings...</div>', unsafe_allow_html=True)
                    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

                    st.markdown('<div class="step">🗄️ Building vector store...</div>', unsafe_allow_html=True)
                    vectordb = Chroma.from_texts(texts=texts, embedding=embedder)

                    st.session_state.retriever    = vectordb.as_retriever(search_kwargs={"k": TOP_K})
                    st.session_state.llm          = ChatOpenAI(model=MODEL, temperature=0, api_key=OPENAI_API_KEY)
                    st.session_state.all_texts    = texts
                    st.session_state.chunks_count = len(texts)
                    st.session_state.pages_count  = len(docs)
                    st.session_state.pdf_name     = uploaded_file.name
                    st.session_state.messages     = []

                    os.unlink(tmp_path)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ {str(e)}")

# ─── STATS ────────────────────────────────────────────────────────────────────
if st.session_state.retriever:
    name = st.session_state.pdf_name or "—"
    short = (name[:14] + "…") if len(name) > 14 else name
    q_count = len(st.session_state.messages) // 2

    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-num">📄</div><div class="stat-lbl">{short}</div></div>
        <div class="stat-card"><div class="stat-num">{st.session_state.pages_count}</div><div class="stat-lbl">Pages</div></div>
        <div class="stat-card"><div class="stat-num">{st.session_state.chunks_count}</div><div class="stat-lbl">Chunks</div></div>
        <div class="stat-card"><div class="stat-num">{q_count}</div><div class="stat-lbl">Asked</div></div>
    </div>
    """, unsafe_allow_html=True)

# ─── CHAT ─────────────────────────────────────────────────────────────────────
if st.session_state.retriever:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">✨</div>
            <div class="empty-title">Your PDF is ready</div>
            <div class="empty-sub">Type a question below and let Advanced RAG do the magic</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-feed">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble-user">{msg["content"]}</div>
                    <div class="av-user">🧑</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if "steps" in msg:
                    s = msg["steps"]
                    scores_str = " · ".join(str(x) for x in s.get("scores", []))
                    st.markdown(f"""
                    <div class="rag-badge">
                        <span class="rag-tag">✍️ rewritten</span>
                        <span class="rag-tag">🔀 {s.get("hybrid_count",0)} chunks</span>
                        <span class="rag-tag">📊 {scores_str}</span>
                        <div class="rewritten-q">"{s.get("rewritten","")}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="msg-bot">
                    <div class="av-bot">📌</div>
                    <div class="bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
                if "sources" in msg:
                    with st.expander("📎 View source chunks"):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f'<div class="chunk">#{i+1} &nbsp;→&nbsp; {src[:260]}…</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── INPUT BAR ──
    st.markdown('<div class="input-wrap"><div class="input-inner">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([7, 1.4, 1])
    with col1:
        question = st.text_input("", placeholder="Ask anything about your PDF...",
                                 label_visibility="collapsed", key="q")
    with col2:
        ask_btn = st.button("Ask →", use_container_width=True)
    with col3:
        clear_btn = st.button("🗑️", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if ask_btn and question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer, sources, steps = advanced_rag(
                    question,
                    st.session_state.retriever,
                    st.session_state.all_texts,
                    st.session_state.llm
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [d.page_content for d in sources],
                    "steps": steps
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
                    "sources": []
                })
        st.rerun()

    if clear_btn:
        st.session_state.messages = []
        st.rerun()