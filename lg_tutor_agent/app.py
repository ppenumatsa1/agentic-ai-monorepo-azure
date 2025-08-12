# micro_tutor_langgraph.py

import os
import json
import logging
from typing import TypedDict, Dict, Any, List
from dotenv import load_dotenv

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

# -------------------- Logging --------------------

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "micro_tutor.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)

# -------------------- Azure OpenAI --------------------
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL_NAME")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    model=AZURE_MODEL,
    temperature=0,
)


# -------------------- Graph State --------------------
class State(TypedDict, total=False):
    # Inputs
    learner_id: str
    topic: str
    question: str
    user_answer: str

    # Control
    attempts: int
    is_correct: bool

    # Outputs
    hint_history: List[str]
    final_explanation: str
    praise: str
    feedback: str  # "Yes" | "No"
    error: str


# -------------------- Nodes --------------------
def select_item(state: State) -> State:
    """Choose the current item based on session index and topic."""
    try:
        topic = state.get("topic") or "arithmetic"
        # Only arithmetic supported for now
        if "question" not in st.session_state or st.session_state.get(
            "new_question", True
        ):
            prev_qs = st.session_state.previous_questions
            prompt = (
                "You are a math tutor. Generate a single arithmetic question for practice. "
                "Do not repeat any of these previous questions: "
                f"{prev_qs}. Return only the question as a string."
            )
            sys = SystemMessage(content=prompt)
            # Use higher temperature for more variety
            varied_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION,
                model=AZURE_MODEL,
                temperature=0.7,
            )
            ai = varied_llm.invoke([sys])
            question = ai.content.strip()
            st.session_state.question = question
            st.session_state.new_question = False
            # Track previous questions
            st.session_state.previous_questions.append(question)

        state["question"] = st.session_state.question
        state.setdefault("attempts", 0)
        state.setdefault("hint_history", [])
        logging.info(f"Selected question: {state['question']}")
    except Exception as e:
        state["error"] = f"select_item error: {e}"
        logging.exception(state["error"])
    return state


def check_answer(state: State) -> State:
    """Ask the model to judge correctness concisely using the canonical answer."""
    try:
        question = state.get("question", "")
        student = (state.get("user_answer") or "").strip()
        sys = SystemMessage(
            content=(
                "You are a concise math grader. Judge the student's answer to the following arithmetic question. "
                "Return JSON ONLY with keys: is_correct (true/false), brief_reason (<=20 words)."
            )
        )
        human = HumanMessage(
            content=json.dumps({"question": question, "student_answer": student})
        )
        ai = llm.invoke([sys, human])
        parsed = {}
        try:
            parsed = json.loads(ai.content)
        except Exception:
            parsed = {
                "is_correct": False,
                "brief_reason": "Could not parse LLM response.",
            }
        state["is_correct"] = bool(parsed.get("is_correct", False))
        state["brief_reason"] = parsed.get("brief_reason", "")
        logging.info(f"Grader result: {parsed}")
    except Exception as e:
        state["error"] = f"check_answer error: {e}"
        logging.exception(state["error"])
    return state


def give_hint(state: State) -> State:
    """Generate a short hint (1 sentence) without revealing the full solution."""
    try:
        question = state.get("question", "")
        student = (state.get("user_answer") or "").strip()
        sys = SystemMessage(
            content=(
                "You are a math tutor. Give ONE short hint (max 1 sentence) for the following arithmetic question. "
                "Do NOT reveal the final answer. Keep it actionable."
            )
        )
        human = HumanMessage(content=f"Question: {question}\nStudent answer: {student}")
        ai = llm.invoke([sys, human])
        hint = ai.content.strip()
        state["hint_history"] = (state.get("hint_history") or []) + [hint]
        state["attempts"] = int(state.get("attempts", 0)) + 1
        logging.info(f"Hint generated. Attempts now {state['attempts']}.")
    except Exception as e:
        state["error"] = f"give_hint error: {e}"
        logging.exception(state["error"])
    return state


def explain(state: State) -> State:
    """Give a concise worked solution referencing the canonical answer."""
    try:
        question = state.get("question", "")
        sys = SystemMessage(
            content=(
                "You are a math tutor. Provide a concise worked solution in <=3 steps for the following arithmetic question, "
                "then state the final answer."
            )
        )
        human = HumanMessage(content=f"Question: {question}")
        ai = llm.invoke([sys, human])
        state["final_explanation"] = ai.content.strip()
        logging.info("Final explanation generated.")
    except Exception as e:
        state["error"] = f"explain error: {e}"
        logging.exception(state["error"])
    return state


def summarize_success(state: State) -> State:
    """Brief praise + key point when correct."""
    try:
        question = state.get("question", "")
        sys = SystemMessage(
            content="You are a supportive tutor. Reply with 1 sentence: praise + key takeaway for the following arithmetic question."
        )
        human = HumanMessage(content=f"Question: {question}")
        ai = llm.invoke([sys, human])
        state["praise"] = ai.content.strip()
        logging.info("Success summary generated.")
    except Exception as e:
        state["error"] = f"summarize_success error: {e}"
        logging.exception(state["error"])
    return state


def route_after_check(state: State) -> str:
    """Router for conditional edges after check_answer."""
    if state.get("error"):
        return "explain"
    is_correct = bool(state.get("is_correct"))
    attempts = int(state.get("attempts", 0))
    if is_correct:
        return "summarize_success"
    # if not correct: give up after 2 tries -> explain
    return "give_hint" if attempts < 2 else "explain"


# -------------------- Build Graph --------------------
workflow = StateGraph(state_schema=State)  # state_schema is required.
workflow.add_node("select_item", select_item)
workflow.add_node("check_answer", check_answer)
workflow.add_node("give_hint", give_hint)
workflow.add_node("explain", explain)
workflow.add_node("summarize_success", summarize_success)

workflow.add_edge(START, "select_item")
workflow.add_edge("select_item", "check_answer")
workflow.add_conditional_edges(  # conditional branching API. :contentReference[oaicite:3]{index=3}
    "check_answer",
    route_after_check,
    {
        "summarize_success": "summarize_success",
        "give_hint": "give_hint",
        "explain": "explain",
    },
)
# End conditions
workflow.add_edge("summarize_success", END)
workflow.add_edge("give_hint", END)  # UI will gather a new attempt and re-run
workflow.add_edge("explain", END)

tutor_app = workflow.compile()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🧮 Math-Tutor (LangGraph)", layout="wide")
st.title("🧮 Math-Tutor (LangGraph + Azure OpenAI)")

# Session init
if "topic" not in st.session_state:
    st.session_state.topic = "arithmetic"
if "item_index" not in st.session_state:
    st.session_state.item_index = 0
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "hint_history" not in st.session_state:
    st.session_state.hint_history = []
if "show_next" not in st.session_state:
    st.session_state.show_next = False
if "previous_questions" not in st.session_state:
    st.session_state.previous_questions = []

col1, col2 = st.columns([2, 1])
with col1:
    st.selectbox("Topic", ["arithmetic"], key="topic", disabled=True)
    # Current question
    if "question" not in st.session_state or st.session_state.get("new_question", True):
        st.session_state.new_question = True
        prev_qs = st.session_state.previous_questions
        prompt = (
            "You are a math tutor. Generate a single arithmetic question for practice. "
            "Do not repeat any of these previous questions: "
            f"{prev_qs}. Return only the question as a string."
        )
        sys = SystemMessage(content=prompt)
        # Use higher temperature for more variety
        varied_llm = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            model=AZURE_MODEL,
            temperature=0.7,
        )
        ai = varied_llm.invoke([sys])
        question = ai.content.strip()
        st.session_state.question = question
        st.session_state.new_question = False
        # Track previous questions
        st.session_state.previous_questions.append(question)
    st.markdown(f"### Question\n{st.session_state.question}")

    # Generate a unique key for the text input to force it to reset
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = 0
    user_answer = st.text_input(
        "Your answer ", key=f"answer_{st.session_state.answer_key}"
    )
    # Display current attempts and hints if any
    if st.session_state.attempts > 0:
        st.write(f"Attempts (this item): **{st.session_state.attempts}**")
        if st.session_state.hint_history:
            st.markdown("**Hints so far:**")
            for idx, hint in enumerate(st.session_state.hint_history, 1):
                st.write(f"{idx}. {hint}")

    submit = st.button("Submit")

    # Display last result if exists
    if "last_result" in st.session_state:
        result_type = st.session_state.last_result["type"]
        if result_type == "correct":
            st.success(st.session_state.last_result["message"])
            if st.session_state.last_result["praise"]:
                st.info(st.session_state.last_result["praise"])
        elif result_type == "explanation":
            st.warning(st.session_state.last_result["message"])
            st.write(st.session_state.last_result["explanation"])
        elif result_type == "hint":
            st.error(st.session_state.last_result["message"])
            if st.session_state.last_result["hint"]:
                st.info(f"💡 {st.session_state.last_result['hint']}")
    if st.session_state.show_next:
        st.divider()
        if st.button("Next Question"):
            # Reset all state for next question
            st.session_state.new_question = True
            if "question" in st.session_state:
                del st.session_state["question"]
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.session_state.attempts = 0
            st.session_state.hint_history = []
            st.session_state.show_next = False
            st.session_state.item_index += 1
            if "answer_key" not in st.session_state:
                st.session_state.answer_key = 0
            st.session_state.answer_key += 1
            st.rerun()

    if submit:
        init_state: State = {
            "learner_id": "demo",
            "topic": st.session_state.topic,
            "question": st.session_state.question,
            "user_answer": user_answer,
            "attempts": st.session_state.attempts,
            "hint_history": st.session_state.hint_history,
        }
        result = tutor_app.invoke(init_state)

        if result.get("error"):
            st.error(result["error"])
        else:
            # Store result in session state
            if result.get("is_correct"):
                st.session_state.last_result = {
                    "type": "correct",
                    "message": "✅ Correct!",
                    "praise": result.get("praise", ""),
                }
                st.session_state.show_next = True
            else:
                # Incorrect path
                if result.get("final_explanation"):
                    st.session_state.last_result = {
                        "type": "explanation",
                        "message": "❌ Not quite. Here's a concise explanation:",
                        "explanation": result.get("final_explanation", ""),
                    }
                    st.session_state.show_next = True
                else:
                    # Hint path
                    brief_reason = result.get(
                        "brief_reason", "Incorrect answer. Try again."
                    )
                    hints = result.get("hint_history", [])
                    latest_hint = hints[-1].replace("Hint: ", "") if hints else None
                    st.session_state.last_result = {
                        "type": "hint",
                        "message": f"❌ Attempt {st.session_state.attempts + 1}: {brief_reason}",
                        "hint": latest_hint,
                    }
                    # Update attempts and hint_history
                    st.session_state.attempts = int(result.get("attempts", 0))
                    st.session_state.hint_history = [
                        h.replace("Hint: ", "") for h in hints
                    ]
            # Add st.rerun() here to ensure UI updates before user can interact further
            st.rerun()

with col2:
    st.markdown("### Progress")
    st.write(f"Topic: **{st.session_state.topic}**")
    st.write(f"Item #: **{st.session_state.item_index + 1}**")

    st.divider()
    feedback = st.radio("Was this helpful?", ["—", "Yes", "No"], index=0)
    if st.button("Submit feedback"):
        if feedback in ("Yes", "No"):
            st.success("Thanks for the feedback!")
        else:
            st.info("Pick Yes/No first.")
