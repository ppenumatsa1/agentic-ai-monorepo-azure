# faq_agent_crewai.py

import os
import sqlite3
import logging
import json
from typing import Dict, Any
from dotenv import load_dotenv

import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# --------------- Setup Logging ---------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "faq_agent_crewai.log")
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
        "LangChain is an open-source framework that simplifies the development of applications using large language models (LLMs), providing modular tools for chains, prompt management, integrations, and agent workflows.",
    ),
    (
        "What is LangGraph?",
        "LangGraph is an open-source, graph-based orchestration framework built on LangChain, designed to manage stateful workflows for complex, multi-step, and multi-agent tasks.",
    ),
    (
        "When should I use LangGraph instead of LangChain?",
        "Use LangGraph when your application requires explicit control flow, branching, loops, human-in-the-loop checkpoints, or persistent state across steps; LangChain is better suited for simpler, linear agent workflows.",
    ),
    (
        "Is LangChain only for Python?",
        "No—while LangChain is widely used in Python, it also has support for JavaScript, enabling developers to build LLM applications across both ecosystems.",
    ),
    (
        "Is LangGraph free to use?",
        "Yes, LangGraph is an open-source, MIT-licensed library that is free to use. Its LangGraph Platform (for scalable deployments) is a separate, optional managed service.",
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


# --------------- FAQ Search Tool ---------------
@tool("search_faq")
def search_faq(question: str) -> str:
    """
    Look up the FAQ database for a question. Return a JSON string:
    {"found": true/false, "answer": "<answer or null>"}.
    Uses simple LIKE matching for demo purposes.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT answer FROM faqs WHERE question LIKE ?", (f"%{question}%",))
        row = cur.fetchone()
        conn.close()
        if row:
            logging.info(f"FAQ match found for '{question}'.")
            return json.dumps({"found": True, "answer": row[0]})
        logging.info(f"No FAQ match found for '{question}'.")
        return json.dumps({"found": False, "answer": None})
    except Exception as e:
        logging.exception(f"search_faq error: {e}")
        return json.dumps({"found": False, "answer": None, "error": str(e)})


# --------------- CrewAI LLM (Azure) ---------------
crew_llm = LLM(
    model=f"azure/{AZURE_MODEL}",
    api_key=AZURE_API_KEY,
    api_base=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# --------------- Agents ---------------
retriever_agent = Agent(
    role="FAQ Retriever",
    goal="Find accurate answers from the local FAQ database using provided tools.",
    backstory="You quickly check a local FAQ list to see if the user's question already has a vetted answer.",
    tools=[search_faq],
    llm=crew_llm,
    allow_delegation=False,
    verbose=False,
)

writer_agent = Agent(
    role="Answer Writer",
    goal="When no exact FAQ is found, write a concise and helpful answer to the user's question.",
    backstory="You craft clear, friendly responses drawing on your general knowledge.",
    llm=crew_llm,
    allow_delegation=False,
    verbose=False,
)

# --------------- Tasks ---------------
search_task = Task(
    description=(
        "Use the search_faq tool to find an answer for the user's question: '{question}'. "
        "Return ONLY JSON with keys: found (true/false) and answer (string or null)."
    ),
    expected_output='JSON: {"found": <bool>, "answer": <string-or-null>}',
    agent=retriever_agent,
    tools=[search_faq],
)

fallback_task = Task(
    description=(
        "You receive the prior task output as context. If it indicates found=true, simply return that answer verbatim. "
        "If found=false, write a concise, friendly answer to the original question: '{question}'. "
        "Return ONLY the final answer text."
    ),
    expected_output="Final answer text only.",
    agent=writer_agent,
)

# --------------- Crew ---------------
crew = Crew(
    agents=[retriever_agent, writer_agent],
    tasks=[search_task, fallback_task],
    process=Process.sequential,  # Search first, then fallback writer
    verbose=False,
)

# --------------- Streamlit UI ---------------
st.set_page_config(page_title="CrewAI FAQ Bot", layout="wide")
st.title("📚 CrewAI FAQ Bot with Feedback")

if "result" not in st.session_state:
    st.session_state.result = None
if "steps" not in st.session_state:
    st.session_state.steps = []

query = st.text_input("Ask a question:")
if st.button("Submit Query"):
    st.session_state.steps = []
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            # Step 1: Kick off crew with question input
            result_text = crew.kickoff(inputs={"question": query})
            # CrewOutput object: extract answer from its attributes if needed
            if hasattr(result_text, "output"):
                final_answer = str(result_text.output).strip()
            else:
                final_answer = str(result_text).strip()

            # Step record: we also log what search_faq returned for transparency
            # (Run tool directly to capture structured step info, mirroring LangGraph steps log)
            raw = search_faq.run(query)  # tool returns string
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"found": False, "answer": None}

            if parsed.get("found"):
                st.session_state.steps.append(
                    {"step": "search_faq", "result": f"FAQ match found for '{query}'."}
                )
            else:
                st.session_state.steps.append(
                    {
                        "step": "search_faq",
                        "result": f"No FAQ match found for '{query}'.",
                    }
                )
                st.session_state.steps.append(
                    {
                        "step": "generate_answer",
                        "result": "Generated fallback answer via LLM.",
                    }
                )

            st.session_state.result = {
                "question": query,
                "answer": final_answer if final_answer else parsed.get("answer") or "",
                "persist_feedback": True,
            }
        except Exception as e:
            logging.exception(e)
            st.session_state.result = {"error": str(e)}

if st.session_state.result:
    result = st.session_state.result
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.markdown(f"**Answer:** {result.get('answer', '_No answer provided_')}")
        # Show interim steps in an expander (like the LangGraph version)
        steps = st.session_state.get("steps", [])
        if steps:
            with st.expander("Show Interim Steps"):
                for i, step in enumerate(steps):
                    st.markdown(f"**Step {i+1}: {step['step']}**")
                    st.write(step["result"])

        feedback = st.radio("Was this helpful?", ["Yes", "No"], key="fb")
        if st.button("Submit Feedback", key="fb_btn"):
            result["feedback"] = feedback
            if result.get("persist_feedback"):
                # Perform DB write here (same as LangGraph version)
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
                # Lightweight regeneration path (call writer agent alone)
                try:
                    regen_task = Task(
                        description=(
                            f"The previous answer wasn't helpful. Rewrite a clearer, more concise answer to: '{result.get('question')}'. "
                            "Return only the final answer text."
                        ),
                        expected_output="Final answer text only.",
                        agent=writer_agent,
                    )
                    regen_crew = Crew(
                        agents=[writer_agent],
                        tasks=[regen_task],
                        process=Process.sequential,
                        verbose=False,
                    )
                    better = regen_crew.kickoff()
                    st.markdown(f"**LLM Answer:** {better}")
                except Exception as e:
                    st.error(f"Regeneration error: {e}")
            else:
                st.success("Thank you for your feedback!")
