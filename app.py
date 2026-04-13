# import os
# import re
# import time
# import streamlit as st
# import psycopg2
# import pandas as pd
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# # ── Config ────────────────────────────────────────────────────────────────────
# NEON_CONNECTION_STRING = os.getenv("NEON_CONNECTION_STRING")
# GEMINI_API_KEY         = os.getenv("GEMINI_API_KEY")

# genai.configure(api_key=GEMINI_API_KEY)

# # ── DB helpers ────────────────────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Connecting to Neon DB…")
# def get_connection():
#     return psycopg2.connect(NEON_CONNECTION_STRING, sslmode="require")

# def run_query(sql: str) -> pd.DataFrame:
#     conn = get_connection()
#     try:
#         return pd.read_sql_query(sql, conn)
#     except Exception:
#         conn.rollback()
#         raise

# # ── Schema introspection ──────────────────────────────────────────────────────
# @st.cache_data(show_spinner="Reading your schema…")
# def introspect_schema() -> list[dict]:
#     """
#     Returns a list of dicts, one per table:
#       { table, columns: [{name, type, nullable, pk}], foreign_keys: [...] }
#     """
#     conn = get_connection()
#     cur  = conn.cursor()

#     cur.execute("""
#         SELECT table_name
#         FROM information_schema.tables
#         WHERE table_schema = 'public'
#           AND table_type   = 'BASE TABLE'
#         ORDER BY table_name;
#     """)
#     tables = [r[0] for r in cur.fetchall()]

#     schema = []
#     for tbl in tables:
#         # columns
#         cur.execute("""
#             SELECT c.column_name,
#                    c.data_type,
#                    c.is_nullable,
#                    CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_pk
#             FROM information_schema.columns c
#             LEFT JOIN (
#                 SELECT ku.column_name
#                 FROM information_schema.table_constraints tc
#                 JOIN information_schema.key_column_usage ku
#                   ON tc.constraint_name = ku.constraint_name
#                  AND tc.table_name      = ku.table_name
#                 WHERE tc.constraint_type = 'PRIMARY KEY'
#                   AND tc.table_name      = %s
#             ) pk ON c.column_name = pk.column_name
#             WHERE c.table_name = %s
#               AND c.table_schema = 'public'
#             ORDER BY c.ordinal_position;
#         """, (tbl, tbl))
#         columns = [
#             {"name": r[0], "type": r[1], "nullable": r[2], "pk": r[3]}
#             for r in cur.fetchall()
#         ]

#         # foreign keys
#         cur.execute("""
#             SELECT kcu.column_name,
#                    ccu.table_name  AS foreign_table,
#                    ccu.column_name AS foreign_column
#             FROM information_schema.table_constraints tc
#             JOIN information_schema.key_column_usage kcu
#               ON tc.constraint_name = kcu.constraint_name
#             JOIN information_schema.constraint_column_usage ccu
#               ON ccu.constraint_name = tc.constraint_name
#             WHERE tc.constraint_type = 'FOREIGN KEY'
#               AND tc.table_name      = %s;
#         """, (tbl,))
#         fks = [
#             {"column": r[0], "ref_table": r[1], "ref_column": r[2]}
#             for r in cur.fetchall()
#         ]

#         schema.append({"table": tbl, "columns": columns, "foreign_keys": fks})

#     cur.close()
#     return schema

# def schema_to_text(entry: dict) -> str:
#     """Convert one table's schema dict to a rich text chunk for embedding."""
#     cols = ", ".join(
#         f"{c['name']} ({c['type']}{'  PK' if c['pk'] else ''})"
#         for c in entry["columns"]
#     )
#     fk_str = ""
#     if entry["foreign_keys"]:
#         fk_str = " | FK: " + ", ".join(
#             f"{f['column']}→{f['ref_table']}.{f['ref_column']}"
#             for f in entry["foreign_keys"]
#         )
#     return f"Table: {entry['table']} | Columns: {cols}{fk_str}"

# # ── FAISS index ───────────────────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Building schema embeddings (first run only)…")
# def build_faiss_index(_schema: list[dict]):
#     """
#     Embeds every table schema and builds a FAISS flat-L2 index.
#     Returns (index, model, texts).
#     _schema prefixed with _ so Streamlit doesn't try to hash it.
#     """
#     model  = SentenceTransformer("all-MiniLM-L6-v2")   # fast, good quality
#     texts  = [schema_to_text(s) for s in _schema]
#     vecs   = model.encode(texts, show_progress_bar=False).astype("float32")
#     dim    = vecs.shape[1]
#     index  = faiss.IndexFlatL2(dim)
#     index.add(vecs)
#     return index, model, texts

# def retrieve_top_k(question: str, index, model, texts: list[str], k: int = 6) -> list[str]:
#     q_vec = model.encode([question]).astype("float32")
#     _, indices = index.search(q_vec, k)
#     return [texts[i] for i in indices[0]]

# # ── SQL validation ────────────────────────────────────────────────────────────
# BLOCKED = re.compile(
#     r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|EXEC)\b",
#     re.IGNORECASE
# )

# def validate_sql(sql: str) -> tuple[bool, str]:
#     if BLOCKED.search(sql):
#         return False, "Only SELECT queries are allowed."
#     if not sql.strip().upper().startswith("SELECT"):
#         return False, "Query must start with SELECT."
#     return True, ""

# # ── Gemini SQL generation ─────────────────────────────────────────────────────
# def build_prompt(question: str, schema_chunks: list[str]) -> str:
#     schema_block = "\n".join(f"  - {c}" for c in schema_chunks)
#     return f"""You are an expert PostgreSQL analyst for a trading organization.

# Relevant table schemas (retrieved from a vector index):
# {schema_block}

# Rules:
# 1. Output ONLY a single valid PostgreSQL SELECT statement — no explanation, no markdown, no backticks.
# 2. Use only the tables listed above.
# 3. Use explicit column names, never SELECT *.
# 4. Add LIMIT 500 unless the query is an aggregate.
# 5. If the question cannot be answered with the given tables, reply exactly: CANNOT_ANSWER

# Question: {question}

# SQL:"""

# def generate_sql(question: str, schema_chunks: list[str], retry_error: str = "") -> str:
#     prompt = build_prompt(question, schema_chunks)
#     if retry_error:
#         prompt += f"\n\nThe previous attempt failed with: {retry_error}\nFix the SQL and try again."

#     model    = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     sql      = response.text.strip().strip("```sql").strip("```").strip()
#     return sql

# def explain_results(question: str, sql: str, df: pd.DataFrame) -> str:
#     preview   = df.head(5).to_markdown(index=False) if not df.empty else "No rows returned."
#     model     = genai.GenerativeModel("gemini-2.0-flash")
#     response  = model.generate_content(
#         f"A trader asked: '{question}'\n"
#         f"The SQL query ran: {sql}\n"
#         f"Result preview:\n{preview}\n\n"
#         f"Write a clear 2-3 sentence plain-English summary of what this result means for the trader."
#     )
#     return response.text.strip()

# # ── Streamlit UI ──────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Trading DB Chat",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Trading Database — Natural Language Query")
# st.caption("Ask questions about your trading data in plain English. Powered by Gemini + FAISS + Neon DB.")

# # Sidebar — connection info & schema explorer
# with st.sidebar:
#     st.header("Connection")
#     if NEON_CONNECTION_STRING:
#         st.success("Neon DB connected")
#     else:
#         st.error("NEON_CONNECTION_STRING not set in .env")

#     st.divider()
#     st.header("Schema explorer")
#     schema = introspect_schema()
#     for entry in schema:
#         with st.expander(f"🗂 {entry['table']} ({len(entry['columns'])} cols)"):
#             for col in entry["columns"]:
#                 pk_tag = " 🔑" if col["pk"] else ""
#                 st.markdown(f"`{col['name']}` — {col['type']}{pk_tag}")
#             if entry["foreign_keys"]:
#                 st.markdown("**Foreign keys:**")
#                 for fk in entry["foreign_keys"]:
#                     st.markdown(f"→ `{fk['column']}` → `{fk['ref_table']}.{fk['ref_column']}`")

# # Build FAISS index (cached after first run)
# index, embed_model, texts = build_faiss_index(tuple(
#     (e["table"], schema_to_text(e)) for e in schema
# ) and schema)

# # Chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "dataframe" in msg:
#             st.dataframe(msg["dataframe"], use_container_width=True)
#         if "sql" in msg:
#             with st.expander("View SQL"):
#                 st.code(msg["sql"], language="sql")

# # Input
# question = st.chat_input("e.g. Show me all trades from last week above $50,000")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Retrieving relevant tables…"):
#             chunks = retrieve_top_k(question, index, embed_model, texts, k=6)

#         sql = None
#         df  = None
#         answer = ""
#         last_error = ""

#         # Generate + self-correct loop (max 3 attempts)
#         for attempt in range(3):
#             with st.spinner(f"Generating SQL (attempt {attempt+1})…"):
#                 sql = generate_sql(question, chunks, retry_error="" if attempt == 0 else last_error)

#             if sql == "CANNOT_ANSWER":
#                 answer = "I couldn't find tables relevant enough to answer this question. Try rephrasing or check the schema explorer on the left."
#                 break

#             ok, reason = validate_sql(sql)
#             if not ok:
#                 last_error = reason
#                 st.toast(f"SQL Validation failed. Retrying in 15s... ({reason})")
#                 time.sleep(15)  # Buffer to protect Free Tier limits
#                 continue

#             try:
#                 df = run_query(sql)
#                 with st.spinner("Summarising results…"):
#                     time.sleep(2)  # Small buffer before hitting the API again
#                     answer = explain_results(question, sql, df)
#                 break
#             except Exception as e:
#                 last_error = str(e)
#                 df = None
#                 st.toast(f"DB Execution failed. Retrying in 15s... ({last_error})")
#                 time.sleep(15)  # Buffer to protect Free Tier limits
#         else:
#             answer = f"Could not generate a working query after 3 attempts. Last error: `{last_error}`"

#         st.markdown(answer)
#         if df is not None and not df.empty:
#             st.dataframe(df, use_container_width=True)
#         if sql and sql != "CANNOT_ANSWER":
#             with st.expander("View generated SQL"):
#                 st.code(sql, language="sql")

#         msg = {"role": "assistant", "content": answer}
#         if df is not None:
#             msg["dataframe"] = df
#         if sql:
#             msg["sql"] = sql
#         st.session_state.messages.append(msg)

import os
import re
import time
import streamlit as st
import psycopg2
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
NEON_CONNECTION_STRING = os.getenv("NEON_CONNECTION_STRING")
OPENROUTER_API_KEY     = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI client pointed at OpenRouter's free tier
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Using Meta's Llama 3.1 8B Instruct model (100% Free on OpenRouter)
# MODEL_NAME = "meta-llama/llama-3.1-8b-instruct:free"
# MODEL_NAME = "google/gemma-2-9b-it:free"
# Replace this:
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"

# With one of these currently available free models:



# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Neon DB…")
def get_connection():
    return psycopg2.connect(NEON_CONNECTION_STRING, sslmode="require")

def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        conn.rollback()
        raise

# ── Schema introspection ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Reading your schema…")
def introspect_schema() -> list[dict]:
    """
    Returns a list of dicts, one per table:
      { table, columns: [{name, type, nullable, pk}], foreign_keys: [...] }
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type   = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    schema = []
    for tbl in tables:
        # columns
        cur.execute("""
            SELECT c.column_name,
                   c.data_type,
                   c.is_nullable,
                   CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_pk
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                  ON tc.constraint_name = ku.constraint_name
                 AND tc.table_name      = ku.table_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_name      = %s
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_name = %s
              AND c.table_schema = 'public'
            ORDER BY c.ordinal_position;
        """, (tbl, tbl))
        columns = [
            {"name": r[0], "type": r[1], "nullable": r[2], "pk": r[3]}
            for r in cur.fetchall()
        ]

        # foreign keys
        cur.execute("""
            SELECT kcu.column_name,
                   ccu.table_name  AS foreign_table,
                   ccu.column_name AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name      = %s;
        """, (tbl,))
        fks = [
            {"column": r[0], "ref_table": r[1], "ref_column": r[2]}
            for r in cur.fetchall()
        ]

        schema.append({"table": tbl, "columns": columns, "foreign_keys": fks})

    cur.close()
    return schema

def schema_to_text(entry: dict) -> str:
    """Convert one table's schema dict to a rich text chunk for embedding."""
    cols = ", ".join(
        f"{c['name']} ({c['type']}{'  PK' if c['pk'] else ''})"
        for c in entry["columns"]
    )
    fk_str = ""
    if entry["foreign_keys"]:
        fk_str = " | FK: " + ", ".join(
            f"{f['column']}→{f['ref_table']}.{f['ref_column']}"
            for f in entry["foreign_keys"]
        )
    return f"Table: {entry['table']} | Columns: {cols}{fk_str}"

# ── FAISS index ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building schema embeddings (first run only)…")
def build_faiss_index(_schema: list[dict]):
    """
    Embeds every table schema and builds a FAISS flat-L2 index.
    Returns (index, model, texts).
    _schema prefixed with _ so Streamlit doesn't try to hash it.
    """
    model  = SentenceTransformer("all-MiniLM-L6-v2")   # fast, good quality
    texts  = [schema_to_text(s) for s in _schema]
    vecs   = model.encode(texts, show_progress_bar=False).astype("float32")
    dim    = vecs.shape[1]
    index  = faiss.IndexFlatL2(dim)
    index.add(vecs)
    return index, model, texts

def retrieve_top_k(question: str, index, model, texts: list[str], k: int = 6) -> list[str]:
    q_vec = model.encode([question]).astype("float32")
    _, indices = index.search(q_vec, k)
    return [texts[i] for i in indices[0]]

# ── SQL validation ────────────────────────────────────────────────────────────
BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|EXEC)\b",
    re.IGNORECASE
)

def validate_sql(sql: str) -> tuple[bool, str]:
    if BLOCKED.search(sql):
        return False, "Only SELECT queries are allowed."
    if not sql.strip().upper().startswith("SELECT"):
        return False, "Query must start with SELECT."
    return True, ""

# ── OpenRouter Generation ─────────────────────────────────────────────────────
def build_prompt(question: str, schema_chunks: list[str]) -> str:
    schema_block = "\n".join(f"  - {c}" for c in schema_chunks)
    return f"""You are an expert PostgreSQL analyst for a trading organization.

Relevant table schemas (retrieved from a vector index):
{schema_block}

Rules:
1. Output ONLY a single valid PostgreSQL SELECT statement — no explanation, no markdown, no backticks.
2. Use only the tables listed above.
3. Use explicit column names, never SELECT *.
4. Add LIMIT 500 unless the query is an aggregate.
5. If the question cannot be answered with the given tables, reply exactly: CANNOT_ANSWER

Question: {question}

SQL:"""

def generate_sql(question: str, schema_chunks: list[str], retry_error: str = "") -> str:
    prompt = build_prompt(question, schema_chunks)
    if retry_error:
        prompt += f"\n\nThe previous attempt failed with: {retry_error}\nFix the SQL and try again."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, # Low temperature for more deterministic code generation
    )
    
    sql = response.choices[0].message.content.strip().strip("```sql").strip("```").strip()
    return sql

def explain_results(question: str, sql: str, df: pd.DataFrame) -> str:
    preview   = df.head(5).to_markdown(index=False) if not df.empty else "No rows returned."
    
    prompt = (
        f"A trader asked: '{question}'\n"
        f"The SQL query ran: {sql}\n"
        f"Result preview:\n{preview}\n\n"
        f"Write a clear 2-3 sentence plain-English summary of what this result means for the trader."
    )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    return response.choices[0].message.content.strip()

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading DB Chat",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Trading Database — Natural Language Query")
st.caption("Ask questions about your trading data in plain English. Powered by OpenRouter + FAISS + Neon DB.")

# Sidebar — connection info & schema explorer
with st.sidebar:
    st.header("Connection")
    if NEON_CONNECTION_STRING:
        st.success("Neon DB connected")
    else:
        st.error("NEON_CONNECTION_STRING not set in .env")
        
    if OPENROUTER_API_KEY:
        st.success("OpenRouter connected")
    else:
        st.error("OPENROUTER_API_KEY not set in .env")

    st.divider()
    st.header("Schema explorer")
    schema = introspect_schema()
    for entry in schema:
        with st.expander(f"🗂 {entry['table']} ({len(entry['columns'])} cols)"):
            for col in entry["columns"]:
                pk_tag = " 🔑" if col["pk"] else ""
                st.markdown(f"`{col['name']}` — {col['type']}{pk_tag}")
            if entry["foreign_keys"]:
                st.markdown("**Foreign keys:**")
                for fk in entry["foreign_keys"]:
                    st.markdown(f"→ `{fk['column']}` → `{fk['ref_table']}.{fk['ref_column']}`")

# Build FAISS index (cached after first run)
index, embed_model, texts = build_faiss_index(tuple(
    (e["table"], schema_to_text(e)) for e in schema
) and schema)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg:
            st.dataframe(msg["dataframe"], use_container_width=True)
        if "sql" in msg:
            with st.expander("View SQL"):
                st.code(msg["sql"], language="sql")

# Input
question = st.chat_input("e.g. Show me all trades from last week above $50,000")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant tables…"):
            chunks = retrieve_top_k(question, index, embed_model, texts, k=6)

        sql = None
        df  = None
        answer = ""
        last_error = ""

        # Generate + self-correct loop (max 3 attempts)
        for attempt in range(3):
            with st.spinner(f"Generating SQL (attempt {attempt+1})…"):
                try:
                    sql = generate_sql(question, chunks, retry_error="" if attempt == 0 else last_error)
                except Exception as e:
                    last_error = f"API Error: {str(e)}"
                    time.sleep(2) # Brief pause on API error
                    continue

            if sql == "CANNOT_ANSWER":
                answer = "I couldn't find tables relevant enough to answer this question. Try rephrasing or check the schema explorer on the left."
                break

            ok, reason = validate_sql(sql)
            if not ok:
                last_error = reason
                st.toast(f"SQL Validation failed. Retrying... ({reason})")
                continue

            try:
                df = run_query(sql)
                with st.spinner("Summarising results…"):
                    answer = explain_results(question, sql, df)
                break
            except Exception as e:
                last_error = str(e)
                df = None
                st.toast(f"DB Execution failed. Retrying... ({last_error})")
        else:
            answer = f"Could not generate a working query after 3 attempts. Last error: `{last_error}`"

        st.markdown(answer)
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
        if sql and sql != "CANNOT_ANSWER":
            with st.expander("View generated SQL"):
                st.code(sql, language="sql")

        msg = {"role": "assistant", "content": answer}
        if df is not None:
            msg["dataframe"] = df
        if sql:
            msg["sql"] = sql
        st.session_state.messages.append(msg)