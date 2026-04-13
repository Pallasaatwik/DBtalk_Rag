# Trading DB — Natural Language Query (RAG + FAISS + Gemini + Neon)

## What this does
Ask plain-English questions about your PostgreSQL trading database.  
The app auto-reads your schema, embeds it with FAISS, retrieves relevant  
tables per question, generates SQL via Gemini, validates it, runs it on  
Neon DB, and explains the result back to you.

---

## Step 1 — Get your Neon DB connection string

1. Go to https://console.neon.tech
2. Open your project → click **"Connection Details"** (top right)
3. Select **"Connection string"** from the dropdown
4. Copy the string — it looks like:
   ```
   postgresql://alice:abc123@ep-cool-name-123.us-east-2.aws.neon.tech/trading_db
   ```
5. Make sure **SSL** is enabled (it is by default on Neon)

---

## Step 2 — Get your Gemini API key

1. Go to https://aistudio.google.com/app/apikey
2. Click **"Create API Key"**
3. Copy the key

---

## Step 3 — Set up the project

```bash
# Clone / download this folder, then:
cd nl_sql_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your credentials
cp .env.example .env
# Open .env and fill in NEON_CONNECTION_STRING and GEMINI_API_KEY
```

---

## Step 4 — Run the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501

---

## How it works (RAG pipeline)

```
Your question
     │
     ▼
Embed question (all-MiniLM-L6-v2)
     │
     ▼
FAISS search → top 6 relevant table schemas
     │
     ▼
Build prompt (schemas + few-shot rules) → Gemini 1.5 Flash
     │
     ▼
SQL validator (blocks non-SELECT, DROP, DELETE, etc.)
     │
     ▼
Execute on Neon DB (read-only)
     │
     ▼  (if error → retry up to 3x with error fed back to Gemini)
     │
     ▼
Gemini explains result in plain English
     │
     ▼
Streamlit displays answer + data table + SQL
```

---

## Project structure

```
nl_sql_rag/
├── app.py              ← full application
├── requirements.txt    ← all dependencies
├── .env.example        ← credential template
└── README.md           ← this file
```

---

## Common issues

| Problem | Fix |
|---|---|
| `SSL required` error | Add `sslmode=require` — already in the code |
| `GEMINI_API_KEY` not found | Make sure `.env` file exists and is filled in |
| First run is slow | Sentence Transformers downloads the model once (~90MB) |
| Table not found in answers | Check the schema explorer sidebar — table must be in `public` schema |
| `No module named faiss` | Use `faiss-cpu` not `faiss` on pip |