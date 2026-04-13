# 🗄️ DB Talk RAG — Natural Language Database Query

Ask questions about your database in plain English.
Powered by OpenRouter (Llama) + FAISS + Neon DB + Streamlit.

## 🚀 Live Demo
👉 **[https://dbtalkrag-7drjbbyvrg3pwcshzsj86w.streamlit.app/](https://dbtalkrag-7drjbbyvrg3pwcshzsj86w.streamlit.app/)**

## 🛠️ Tech Stack
- **LLM**: OpenRouter (Nvidia Nemotron 120B) via OpenAI-compatible API
- **Vector Store**: FAISS (local, fast)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Database**: Neon DB (PostgreSQL)
- **UI**: Streamlit

## ⚙️ How it works
1. Introspects your PostgreSQL schema automatically
2. Embeds all table schemas into a FAISS vector index
3. On each question, retrieves the top relevant tables
4. Sends schema + question to LLM to generate SQL
5. Validates and executes SQL on Neon DB
6. Returns results + plain-English explanation

## 🔧 Run locally
```bash
pip install -r requirements.txt
# Add your keys to .env
streamlit run app.py
```

## 🔑 Environment Variables
```
NEON_CONNECTION_STRING=your_neon_connection_string
OPENROUTER_API_KEY=your_openrouter_key
```
