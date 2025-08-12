# LangGraph FAQ Bot

A Streamlit-powered FAQ bot that combines local FAQ search with Azure OpenAI fallback, built using LangChain, LangGraph, and SQLite.

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain & LangGraph** (agent workflow)
- **Azure OpenAI** (LLM fallback)
- **SQLite** (FAQ and feedback storage)

## User Flow

1. **search_faq node**: Looks up the user's question in a local SQLite FAQ database.
2. **Branch:**
   - If a match is found (`found == True`), the answer is shown and the user is sent to the feedback node.
   - If no match (`found == False`), the workflow routes to the generate_answer node.
3. **generate_answer node**: Uses Azure OpenAI to generate a fallback answer.
4. **feedback node**: Presents the answer and captures user feedback.
   - If feedback is negative, the bot loops back to generate_answer (or could escalate in future versions).

## How It Works

- Users enter a question in the Streamlit UI.
- The bot first tries to answer from the local FAQ database.
- If no FAQ is found, it generates an answer using Azure OpenAI.
- The user can provide feedback; negative feedback triggers a new LLM answer.

## Setup

### Prerequisites

- Python 3.11 or newer
- [pip](https://pip.pypa.io/en/stable/installation/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/) (recommended)

### Python Environment Setup

#### Windows

```powershell
# Open PowerShell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

1. Set Azure OpenAI credentials in `.env` (see template if provided).

### Run the App

```bash
streamlit run app.py
```

## Directory Structure

- `app.py` — main application and workflow
- `faq.db` — SQLite database for FAQs and feedback
- `.env` — Azure OpenAI credentials
- `requirements.txt` — Python dependencies

---

This project demonstrates a modular, agentic approach to FAQ automation with human-in-the-loop feedback and LLM fallback.
