import os
import sqlite3
import logging
from typing import Optional, List, Dict
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)  # needed when replaying chat history in Streamlit


# --------------- Setup Logging ---------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "todo_agent.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)

# Load environment variables from .env file
load_dotenv()

# 📦 -- Database initialization --
DB_PATH = os.path.join(os.path.dirname(__file__), "tasks.sqlite")
_conn: sqlite3.Connection


def _get_conn() -> sqlite3.Connection:
    global _conn
    try:
        if "_get_conn_conn" not in globals():
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    description  TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    due_date     TEXT,
                    completed_at TEXT
                );
                """
            )
            _conn.commit()
        return _conn
    except Exception as e:
        logging.error(f"Error initializing DB connection: {e}")
        raise


# ✅ -- Task functions --


def add_task(description: str, due_date: Optional[str] = None) -> str:
    """Add a new task with optional due date (YYYY-MM-DD)."""
    try:
        conn = _get_conn()
        created = datetime.utcnow().isoformat()
        cur = conn.execute(
            "INSERT INTO tasks (description, created_at, due_date) VALUES (?, ?, ?)",
            (description.strip(), created, due_date),
        )
        conn.commit()
        logging.info(f"Added task: {description} (Due: {due_date})")
        return f"Task #{cur.lastrowid} added."
    except Exception as e:
        logging.error(f"Error adding task: {e}")
        return f"Error adding task: {e}"


def list_tasks(only_incomplete: bool = False) -> List[Dict]:
    """Return full list of tasks; or incomplete ones if only_incomplete=True."""
    try:
        conn = _get_conn()
        if only_incomplete:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE completed_at IS NULL ORDER BY id"
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
        result = []
        for id_, desc, created, due, comp in rows:
            result.append(
                {
                    "id": id_,
                    "description": desc,
                    "created_at": created,
                    "due_date": due,
                    "completed_at": comp or None,
                }
            )
        logging.info(f"Listed {len(result)} tasks (only_incomplete={only_incomplete})")
        return result
    except Exception as e:
        logging.error(f"Error listing tasks: {e}")
        return []


def complete_task(task_id: int) -> str:
    """Mark a task as completed by its numeric ID."""
    try:
        conn = _get_conn()
        now = datetime.utcnow().isoformat()
        cur = conn.execute(
            "UPDATE tasks SET completed_at = ? WHERE id = ? AND completed_at IS NULL",
            (now, task_id),
        )
        conn.commit()
        if cur.rowcount:
            logging.info(f"Task #{task_id} marked complete.")
            return f"Task #{task_id} marked complete."
        else:
            logging.warning(f"Task #{task_id} not found or already completed.")
            return f"Task #{task_id} not found or already completed."
    except Exception as e:
        logging.error(f"Error completing task: {e}")
        return f"Error completing task: {e}"


def delete_task(task_id: int) -> str:
    """Delete a task by its numeric ID."""
    try:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        if cur.rowcount:
            logging.info(f"Task #{task_id} deleted.")
            return f"Task #{task_id} deleted."
        else:
            logging.warning(f"Task #{task_id} not found.")
            return f"Task #{task_id} not found."
    except Exception as e:
        logging.error(f"Error deleting task: {e}")
        return f"Error deleting task: {e}"


# 🔧 -- Convert functions into StructuredTools --

add_task_tool = StructuredTool.from_function(
    func=add_task,
    name="add_task",
    description="Add a new task with a description and optional due date (YYYY-MM-DD)",
    return_direct=True,
)

list_tasks_tool = StructuredTool.from_function(
    func=list_tasks,
    name="list_tasks",
    description="List all tasks, or optionally only incomplete ones",
)

complete_task_tool = StructuredTool.from_function(
    func=complete_task,
    name="complete_task",
    description="Mark a specific task (by ID) as completed",
    return_direct=True,
)

delete_task_tool = StructuredTool.from_function(
    func=delete_task,
    name="delete_task",
    description="Delete a specific task by its ID",
    return_direct=True,
)

tools = [add_task_tool, list_tasks_tool, complete_task_tool, delete_task_tool]

# 🧠 -- Agent + LLM setup --

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent to‑do list assistant. Use the tools to add, list, complete, or delete tasks.",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_runnable = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# 🖥️ -- Streamlit UI loop --

st.set_page_config(page_title="AI To-Do Agent (StructuredTool)", page_icon="📝")
st.title("📝 AI To‑Do Agent")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input(
    "Your action (e.g. ‘Add buy milk by tomorrow’, ‘Show incomplete tasks’):"
)

if st.button("Submit"):
    if not user_input or not user_input.strip():
        st.error("Please say something.")
    else:
        response = agent_executor.invoke(
            {"input": user_input, "chat_history": st.session_state.history}
        )
        # update memory history
        stored = agent_executor.memory.load_memory_variables({})["chat_history"]
        st.session_state.history = stored
        # show only the newest AI line (last message)
        if stored and isinstance(stored[-1], AIMessage):
            st.success(stored[-1].content)

# Display task list always (optional)
st.markdown("### Current tasks (from DB):")
items = list_tasks(only_incomplete=False)
if not items:
    st.write("_No tasks yet._")
else:
    for t in items:
        comp = "✅ Done" if t["completed_at"] else "🔲 Pending"
        due = f" (Due: {t['due_date']})" if t["due_date"] else ""
        st.write(
            f"- **{t['id']}**: {t['description']}{due} — *Created: {t['created_at']}* — *{comp}*"
        )
