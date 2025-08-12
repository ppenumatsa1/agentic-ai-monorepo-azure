# data_cleaner_agent.py


import os
import re
import logging
from typing import Optional, List, Dict
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import streamlit as st


from pydantic import BaseModel

from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage


# --------------- Setup Logging ---------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "data_wrangler_agent.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)

# Load environment variables from .env file
load_dotenv()


# --- Global DataFrame state ---
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = None


def load_dataframe(file) -> Dict:
    try:
        _, ext = os.path.splitext(file.name.lower())
        if ext in (".csv",):
            df = pd.read_csv(file)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        st.session_state.original_df = df.copy()
        st.session_state.df_global = df.copy()
        logging.info(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} columns.")
        return {"rows": df.shape[0], "columns": df.shape[1]}
    except Exception as e:
        logging.error(f"Error loading dataframe: {e}")
        raise


# --- Cleaning tools ---
def remove_duplicates() -> str:
    try:
        df = st.session_state.df_global
        before = len(df)
        # Strip whitespace from all string columns to ensure true duplicate detection
        str_cols = df.select_dtypes(include=["object"]).columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
        df = df.drop_duplicates().reset_index(drop=True)
        after = len(df)
        st.session_state.df_global = df
        msg = f"Removed {before-after} duplicate rows. DataFrame now has {df.shape[0]} rows, {df.shape[1]} columns."
        logging.info(msg)
        return msg
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return f"Error removing duplicates: {e}"


def standardize_dates(column: str, date_format: str = "%Y-%m-%d") -> str:
    try:
        df = st.session_state.df_global
        if column not in df.columns:
            msg = f"Column '{column}' not found."
            logging.warning(msg)
            return msg
        parsed = pd.to_datetime(df[column], errors="coerce", dayfirst=False)
        current_date = pd.Timestamp.now().strftime(date_format)
        parsed_filled = parsed.fillna(current_date)
        df[column] = pd.to_datetime(parsed_filled).dt.strftime(date_format)
        st.session_state.df_global = df
        msg = f"Standardized all entries in column '{column}' to {date_format}. Empty values set to current date ({current_date}). DataFrame now has {df.shape[0]} rows, {df.shape[1]} columns."
        logging.info(msg)
        return msg
    except Exception as e:
        logging.error(f"Error standardizing dates: {e}")
        return f"Error standardizing dates: {e}"


def extract_emails(source_column: str, target_column: str = "extracted_email") -> str:
    try:
        df = st.session_state.df_global
        if source_column not in df.columns:
            msg = f"Column '{source_column}' not found."
            logging.warning(msg)
            return msg
        emails = (
            df[source_column]
            .astype(str)
            .str.extract(
                r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", expand=False
            )
        )
        df[target_column] = emails
        st.session_state.df_global = df
        count = emails.notna().sum()
        msg = f"Extracted {count} emails into new column '{target_column}'. DataFrame now has {df.shape[0]} rows, {df.shape[1]} columns."
        logging.info(msg)
        return msg
    except Exception as e:
        logging.error(f"Error extracting emails: {e}")
        return f"Error extracting emails: {e}"


def drop_empty_columns(threshold: float = 0.5) -> str:
    try:
        df = st.session_state.df_global
        n = len(df)
        drops = [col for col in df.columns if df[col].isna().sum() / n > threshold]
        df = df.drop(columns=drops)
        st.session_state.df_global = df
        if drops:
            msg = f"Dropped columns: {', '.join(drops)}."
        else:
            msg = "No columns exceeded missing threshold."
        logging.info(msg)
        return msg
    except Exception as e:
        logging.error(f"Error dropping empty columns: {e}")
        return f"Error dropping empty columns: {e}"


def fill_missing(
    column: str, method: str = "constant", value: Optional[str] = None
) -> str:
    try:
        df = st.session_state.df_global
        if column not in df.columns:
            msg = f"Column '{column}' not found."
            logging.warning(msg)
            return msg
        missing_before = df[column].isna().sum()
        if method == "mean":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            fill = df[column].mean()
        elif method == "median":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            fill = df[column].median()
        else:
            fill = value
        df[column] = df[column].fillna(fill)
        st.session_state.df_global = df
        msg = f"Filled {missing_before} missing in '{column}' with {fill}."
        logging.info(msg)
        return msg
    except Exception as e:
        logging.error(f"Error filling missing values: {e}")
        return f"Error filling missing values: {e}"


# --- Wrap tools ---
remove_duplicates_tool = StructuredTool.from_function(
    remove_duplicates,
    name="remove_duplicates",
    description="Remove exact duplicate rows",
    return_direct=True,
)
standardize_dates_tool = StructuredTool.from_function(
    standardize_dates,
    name="standardize_dates",
    description="Standardize dates in a column to ISO format YYYY-MM-DD",
    return_direct=True,
)
extract_emails_tool = StructuredTool.from_function(
    extract_emails,
    name="extract_emails",
    description="Extract emails from a text column to a new column",
    return_direct=True,
)
drop_empty_columns_tool = StructuredTool.from_function(
    drop_empty_columns,
    name="drop_empty_columns",
    description="Drop columns with missing fraction above threshold (float 0-1)",
    return_direct=True,
)
fill_missing_tool = StructuredTool.from_function(
    fill_missing,
    name="fill_missing",
    description="Fill missing values in a column using method: constant, mean, median, specifying value if constant",
    return_direct=True,
)

tools = [
    remove_duplicates_tool,
    standardize_dates_tool,
    extract_emails_tool,
    drop_empty_columns_tool,
    fill_missing_tool,
]

# --- Agent setup ---

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
            "You are a data cleaning assistant. You have tools to remove duplicates, standardize dates, extract emails, drop empty columns and fill missing values.",
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
st.set_page_config(page_title="LLM Data Wrangler Agent", layout="wide")
st.title("📊 Data Wrangler Agent")


uploaded = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xls", "xlsx"])
if uploaded:
    # Only load the file if it's new or not already loaded
    if st.session_state.file_loaded != uploaded.name:
        summary = load_dataframe(uploaded)
        st.session_state.file_loaded = uploaded.name
        st.success(f"Loaded {summary['rows']} rows × {summary['columns']} columns.")

    st.write(f"### Preview (All Rows)")
    st.dataframe(st.session_state.df_global)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Reset to Original Data"):
            st.session_state.df_global = st.session_state.original_df.copy()
            st.success("Data reset to original upload.")
            st.rerun()
    with col2:
        csv = st.session_state.df_global.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned CSV", data=csv, file_name="cleaned.csv", mime="text/csv"
        )

    user_input = st.text_input(
        "Enter cleaning instructions (e.g. 'Remove duplicates and drop empty columns')"
    )

    if st.button("Run"):
        if user_input.strip():
            res = executor.invoke(
                {
                    "input": user_input,
                    "chat_history": st.session_state.get("history", []),
                }
            )
            st.session_state.history = executor.memory.load_memory_variables({})[
                "chat_history"
            ]
            if st.session_state.history and isinstance(
                st.session_state.history[-1], AIMessage
            ):
                st.markdown(f"**Agent:** {st.session_state.history[-1].content}")

        st.write(f"### After Cleaning Preview (All Rows)")
        st.dataframe(st.session_state.df_global)

else:
    st.info("Please upload a CSV or XLSX file to begin.")
