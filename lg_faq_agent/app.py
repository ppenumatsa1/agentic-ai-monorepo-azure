# faq_agent_langgraph.py

import os
import sqlite3
import logging
import json
from pprint import pformat
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd

# --------------- Setup Logging ---------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "faq_agent.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)

# --------------- Load Environment Vars ---------------
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL_NAME")

# --------------- Initialize Database ---------------
DB_PATH = os.path.join(os.path.dirname(__file__), "faq.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            feedback TEXT
        )
    """
    )
    conn.commit()
    conn.close()


init_db()

# --------------- Insert Sample FAQ Data ---------------
SAMPLE_FAQS = [
    (
        "What is LangChain?",
        "LangChain is an open-source framework that simplifies the development of applications using large language models (LLMs), providing modular tools for chains, prompt management, integrations, and agent workflows.",  # Source: :contentReference[oaicite:1]{index=1}
    ),
    (
        "What is LangGraph?",
        "LangGraph is an open-source, graph-based orchestration framework built on LangChain, designed to manage stateful workflows for complex, multi-step, and multi-agent tasks.",  # Source: :contentReference[oaicite:2]{index=2}
    ),
    (
        "When should I use LangGraph instead of LangChain?",
        "Use LangGraph when your application requires explicit control flow, branching, loops, human-in-the-loop checkpoints, or persistent state across steps; LangChain is better suited for simpler, linear agent workflows.",  # Source: :contentReference[oaicite:3]{index=3}
    ),
    (
        "Is LangChain only for Python?",
        "No—while LangChain is widely used in Python, it also has support for JavaScript, enabling developers to build LLM applications across both ecosystems.",  # Source: :contentReference[oaicite:4]{index=4}
    ),
    (
        "Is LangGraph free to use?",
        "Yes, LangGraph is an open-source, MIT‑licensed library that is free to use. Its LangGraph Platform (for scalable deployments) is a separate, optional managed service.",  # Source: :contentReference[oaicite:5]{index=5}
    ),
]


def seed_faqs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for q, a in SAMPLE_FAQS:
        cur.execute("INSERT INTO faqs (question, answer) VALUES (?, ?)", (q, a))
    conn.commit()
    conn.close()


# Only seed if no data
conn = sqlite3.connect(DB_PATH)
count = conn.execute("SELECT COUNT(*) FROM faqs").fetchone()[0]
conn.close()
if count == 0:
    seed_faqs()
    logging.info("Seeded sample FAQ data.")

# --------------- LLM Setup ---------------
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    model=AZURE_MODEL,
    temperature=0,
)


# --------------- Define State Schema ---------------
class State(TypedDict, total=False):
    question: str
    answer: str
    found: bool
    feedback: str
    error: str
    steps: list


# --------------- Node Functions ---------------


def search_faq_node(state: State) -> State:
    try:
        q = state.get("question", "")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT answer FROM faqs WHERE question LIKE ?", (f"%{q}%",))
        res = cur.fetchone()
        conn.close()
        if res:
            state["answer"] = res[0]
            state["found"] = True
            logging.info(f"FAQ match found for '{q}'.")
            step_info = {"step": "search_faq", "result": f"FAQ match found for '{q}'."}
        else:
            state["found"] = False
            logging.info(f"No FAQ match found for '{q}'.")
            step_info = {
                "step": "search_faq",
                "result": f"No FAQ match found for '{q}'.",
            }
        if "steps" not in state:
            state["steps"] = []
        state["steps"].append(step_info)
    except Exception as e:
        state["error"] = f"Search error: {e}"
        logging.error(state["error"])
    return state


def generate_answer_node(state: State) -> State:
    try:
        q = state.get("question", "")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Please answer the following question:\n'{q}'"),
        ]
        response = llm(messages)
        state["answer"] = response.content
        logging.info("Generated fallback answer via LLM.")
        step_info = {
            "step": "generate_answer",
            "result": "Generated fallback answer via LLM.",
        }
        if "steps" not in state:
            state["steps"] = []
        state["steps"].append(step_info)
    except Exception as e:
        state["error"] = f"LLM error: {e}"
        logging.error(state["error"])
    return state


def feedback_node(state: State) -> State:
    # Do not write to DB here; just set a flag
    state["persist_feedback"] = True
    step_info = {
        "step": "feedback",
        "result": f"Feedback ready to be stored: {state.get('feedback')}",
    }
    if "steps" not in state:
        state["steps"] = []
    state["steps"].append(step_info)
    return state


# --------------- Build LangGraph Workflow ---------------

workflow = StateGraph(state_schema=State)
workflow.add_node("search_faq", search_faq_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("feedback", feedback_node)

# Conditional branching based on FAQ match
workflow.add_edge(START, "search_faq")
workflow.add_conditional_edges(
    "search_faq", lambda state: "feedback" if state.get("found") else "generate_answer"
)
workflow.add_edge("generate_answer", "feedback")
workflow.add_edge("feedback", END)

faq_agent = workflow.compile()

# --------------- Streamlit UI ---------------
st.set_page_config(page_title="LangGraph FAQ Bot", layout="wide")
st.title("📚 FAQ Bot with Feedback")

if "question" not in st.session_state:
    st.session_state.question = None
    st.session_state.result = None
    st.session_state.llm_answer = None

query = st.text_input("Ask a question:")
if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.question = query
        init_state: State = {"question": query, "steps": []}
        st.session_state.result = faq_agent.invoke(init_state)
        st.session_state.llm_answer = None

if st.session_state.result:
    result = st.session_state.result
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.markdown(f"**Answer:** {result.get('answer', '_No answer provided_')}")
        # Show all interim steps inside a single expander
        steps = result.get("steps", [])
        if steps:
            with st.expander("Show Interim Steps"):
                for i, step in enumerate(steps):
                    st.markdown(f"**Step {i+1}: {step['step']}**")
                    st.write(step["result"])
        feedback = st.radio("Was this helpful?", ["Yes", "No"], key="fb")
        if st.button("Submit Feedback", key="fb_btn"):
            result["feedback"] = feedback
            if result.get("persist_feedback"):
                # Perform DB write here
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO feedback (question, answer, feedback) VALUES (?, ?, ?)",
                        (
                            result.get("question"),
                            result.get("answer"),
                            result.get("feedback"),
                        ),
                    )
                    conn.commit()
                    conn.close()
                    logging.info("Feedback stored (from UI).")
                except Exception as e:
                    st.error(f"Feedback DB error: {e}")
            if feedback == "No":
                llm_res = generate_answer_node(result)
                st.session_state.llm_answer = llm_res.get(
                    "answer", "_No answer provided_"
                )
            else:
                faq_agent.invoke(result)
                st.success("Thank you for your feedback!")

    if st.session_state.llm_answer:
        st.markdown(f"**LLM Answer:** {st.session_state.llm_answer}")
