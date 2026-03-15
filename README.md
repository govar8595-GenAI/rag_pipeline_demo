# 🧠 AskMyPDF — Advanced RAG Pipeline

> Upload any PDF. Ask anything. Get intelligent answers.

---

## ✨ What is this?

**AskMyPDF** is an Advanced RAG (Retrieval-Augmented Generation) application built with LangChain and Streamlit. It goes beyond basic RAG by adding query rewriting, hybrid search, and re-ranking to deliver more accurate answers from any PDF document.

---

## 🚀 RAG Pipeline

```
User Question
     ↓
✍️  Query Rewriting      →  GPT rewrites the question to improve retrieval
     ↓
🔀  Hybrid Search         →  Semantic (ChromaDB) + Keyword (BM25) combined
     ↓
📊  Re-ranking            →  Scores and filters chunks by relevance
     ↓
🤖  Answer Generation     →  GPT-4o-mini answers from the best chunks
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| Document Loader | PyPDFLoader (LangChain) |
| Text Splitting | RecursiveCharacterTextSplitter |
| Framework | LangChain |
| UI | Streamlit |

---

## 📁 Project Structure

```
langchain_demo/
│
├── rag_app.py          # Main Advanced RAG Streamlit app
├── chroma_demo.py      # Naive RAG demo (basic version)
├── textsplitter.py     # Text splitting experiments
├── chroma_db/          # Local ChromaDB vector store
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repo:**
```bash
git clone https://github.com/govar8595-GenAI/rag_pipeline_demo.git
cd rag_pipeline_demo
```

**2. Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
```

**3. Install dependencies:**
```bash
pip install streamlit langchain langchain-community langchain-openai \
            langchain-huggingface langchain-text-splitters \
            chromadb rank_bm25 pypdf sentence-transformers
```

**4. Add your OpenAI API key in `rag_app.py`:**
```python
OPENAI_API_KEY = "sk-your-key-here"
```

**5. Run the app:**
```bash
streamlit run rag_app.py
```

---

## 🎯 Features

- 📄 **PDF Upload** — Upload any PDF directly in the browser
- ✍️ **Query Rewriting** — Automatically improves your question before searching
- 🔀 **Hybrid Search** — Combines semantic + keyword search for better retrieval
- 📊 **Re-ranking** — Scores chunks by relevance before passing to LLM
- 🧠 **OpenAI Embeddings** — 1536-dimensional vectors for high quality retrieval
- 🎨 **Pinterest UI** — Beautiful warm-toned Streamlit interface
- 🔍 **Source Viewer** — See exactly which chunks were used to answer

---

## 📊 Naive RAG vs Advanced RAG

| Feature | Naive RAG | Advanced RAG |
|---|---|---|
| Query Rewriting | ❌ | ✅ |
| Hybrid Search | ❌ | ✅ |
| Re-ranking | ❌ | ✅ |
| Answer Quality | Good | Much Better |
| Retrieval | Single vector search | BM25 + Semantic |

---

## 🖼️ Screenshots

> Upload your PDF → Process → Ask anything!

---

## 📚 What I Learned

- How RAG pipelines work end to end
- Difference between Naive and Advanced RAG
- How to combine BM25 + semantic search (Hybrid Search)
- How query rewriting improves retrieval accuracy
- How to build production-grade UIs with Streamlit
- Working with LangChain, ChromaDB and OpenAI APIs

---

## 👨‍💻 Author

**Govardhan K**
Batch 6 — Social Eagle Generative AI Course

---

## 📄 License

MIT License
