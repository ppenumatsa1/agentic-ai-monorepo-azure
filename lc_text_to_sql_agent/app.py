# text_to_sql_agent.py

import os
import sqlite3
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

import pandas as pd
import json
import streamlit as st

from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage


# Testing data
# Users Table (users.csv) — Sample Questions
# How many users signed up in March 2025?
# Expected answer: There are 2 users who signed up in March 2025.

# List user IDs and names of people whose name starts with ‘A’ or ‘B’.
# Expected answer: The user IDs and names of people whose names start with 'A' or 'B' are: user ID 1 with the name Alice, and user ID 2 with the name Bob.

# When did user_id = 4 sign up?
# Expected answer: User ID 4 signed up on 2025-03-01.

# Show all users who signed up before February 2025.
# Expected answer: There is one user who signed up before February 2025: Alice, with the email alice@example.com, who signed up on January 15, 2025.

# Count the total number of users in the database.
# Expected answer: The total number of users in the database is 5.

# ------------------------------------------------

# Orders Table (orders.csv) — Sample Questions
# How many orders were placed in March 2025?
# Expected answer: There were 4 orders placed in March 2025.

# What is the total order amount for each user_id?
# Expected answer: The total order amount for user_id 1 is 325.75, for user_id 2 is 300.0, and for user_id 3 is 125.0.

# Show order_id, user_id, and amount for orders over 100.
# Expected answer: There are three orders with amounts over 100: order ID 101 by user ID 1 with an amount of 250.5, order ID 104 by user ID 3 with an amount of 125.0, and order ID 105 by user ID 2 with an amount of 200.0.

# What is the average order amount across all orders?
# Expected answer: The average order amount across all orders is 150.15.

# List all orders placed by user_id = 2.
# Expected answer: User with user_id 2 placed two orders: order_id 102 on 2025-03-07 with an amount of 100.0, and order_id 105 on 2025-04-01 with an amount of 200.0.

# --------------- Setup Logging ---------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "text_to_sql_langgraph.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE_PATH)],
)

# Load Azure credentials
load_dotenv()

# --- SQLite / DB logic ---
if "conn" not in st.session_state:
    st.session_state.conn = None
    st.session_state.table_name = None


def load_csv_to_sqlite(file) -> Dict:
    name = os.path.splitext(file.name)[0]
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df = pd.read_csv(file)
    df.to_sql(name, conn, index=False, if_exists="replace")
    st.session_state.conn = conn
    st.session_state.table_name = name
    nrows, ncols = df.shape
    logging.info(f"Loaded CSV into SQLite table '{name}': {nrows} rows, {ncols} cols")
    return {"table": name, "rows": nrows, "columns": ncols}


def get_schema() -> Dict:
    conn = st.session_state.conn
    if conn is None:
        return {"schema": ""}
    cur = conn.execute("PRAGMA table_info(%s)" % st.session_state.table_name)
    cols = [row[1] for row in cur.fetchall()]
    schema = {st.session_state.table_name: cols}
    # Return schema as JSON string to avoid non-string AIMessage content
    return json.dumps({"schema": schema})


def execute_query(query: str) -> Dict:
    conn = st.session_state.conn
    if conn is None:
        return {"error": "No database loaded."}
    q = query.strip().lower()
    if not q.startswith("select"):
        return {"error": "Only SELECT queries are allowed."}
    # enforce limit
    if "limit" not in q:
        query = query.rstrip(";") + " LIMIT 50;"
    else:
        query = query
    try:
        df = pd.read_sql_query(query, conn)
        rows = df.to_dict(orient="records")
        # Return query results as JSON string to avoid non-string AIMessage content
        return json.dumps({"rows": rows, "columns": list(df.columns)})
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return {"error": str(e)}


# --- LangChain tool wrappers ---
get_schema_tool = StructuredTool.from_function(
    get_schema,
    name="get_schema",
    description="Get table and column names of the database schema",
    return_direct=False,
)

execute_query_tool = StructuredTool.from_function(
    execute_query,
    name="execute_query",
    description="Execute a safe SELECT query (max 50 rows only)",
    return_direct=False,
)

tools = [get_schema_tool, execute_query_tool]

# --- Agent setup ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an SQL assistant. Follow these steps for every question:
            1. Use get_schema() to learn the schema
            2. Write and execute_query() to get data
            3. Format the query results into a natural language response
            4. Always phrase your answer as a complete sentence, for example:
               - "There are 5 users who signed up in March 2025"
               - "User ID 4 signed up on 2025-03-15"
               - "Found 3 users whose names start with 'A'"
               - "The total number of users is 42"
            Never return raw data, JSON, or SQL syntax in your response.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# --- Streamlit UI ---
st.set_page_config(page_title="Text‑to‑SQL Agent", layout="wide")
st.title("💬 Text‑to‑SQL Agent")

uploaded = st.file_uploader("Upload a CSV file (single table)", type=["csv"])
if uploaded:
    if st.session_state.conn is None or st.session_state.uploaded_name != uploaded.name:
        summary = load_csv_to_sqlite(uploaded)
        st.session_state.uploaded_name = uploaded.name
        st.success(
            f"Loaded table '{summary['table']}' with {summary['rows']} rows and {summary['columns']} columns."
        )

    schema_str = get_schema()
    try:
        schema_dict = json.loads(schema_str)
        schema = schema_dict["schema"]
    except Exception:
        schema = {}
    st.markdown("**Schema:**")
    for tbl, cols in schema.items():
        st.write(f"- **{tbl}**: {', '.join(cols)}")

    user_input = st.text_input(
        "Ask a question about this data (e.g. 'How many users?')"
    )
    if st.button("Ask"):
        if user_input.strip():
            # Invoke the agent, which returns a dict with output and intermediate_steps
            result = executor.invoke({"input": user_input})
            st.session_state.history = executor.memory.load_memory_variables({})[
                "chat_history"
            ]
            # Display only the agent's natural language answer
            if isinstance(result, dict) and "output" in result:
                st.markdown(f"**Agent:** {result['output']}")
                # Optionally show intermediate steps for debugging
                if "intermediate_steps" in result:
                    with st.expander("Show intermediate steps"):
                        st.write(result["intermediate_steps"])
            else:
                st.markdown(f"**Agent:** {result}")

else:
    st.info("Please upload a CSV file to begin.")
