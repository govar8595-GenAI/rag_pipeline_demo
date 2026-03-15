# 🧠 AskMyPDF — Advanced RAG Pipeline

> Upload any PDF. Ask anything. Get intelligent answers.


## 🚀 RAG Pipeline
```
User Question → Query Rewriting → Hybrid Search → Re-ranking → Answer
```

## 🛠️ Tech Stack
- LLM: GPT-4o-mini (OpenAI)
- Embeddings: text-embedding-3-small (OpenAI)
- Vector Store: ChromaDB
- Keyword Search: BM25
- Framework: LangChain
- UI: Streamlit

## ⚙️ Setup
1. Clone the repo
2. Install dependencies: `pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters chromadb rank_bm25 pypdf`
3. Add your OpenAI API key in `rag_app.py`
4. Run: `streamlit run rag_app.py`

## 👨‍💻 Author
Govardhan K — Batch 6, Social Eagle Generative AI Course
