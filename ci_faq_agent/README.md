# CrewAI FAQ Bot

A Streamlit-powered FAQ bot that combines local FAQ search with Azure OpenAI fallback, built using CrewAI and SQLite.

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **CrewAI** (agent workflow)
- **Azure OpenAI** (LLM fallback)
- **SQLite** (FAQ and feedback storage)

## User Flow

1. **FAQ Retriever Agent (with search_faq tool and search task)**: Receives the user's question and uses the `search_faq` tool to look up the local SQLite FAQ database.
2. **Task Branching:**
   - If a match is found (`found == True`), the answer is shown and the user is sent to the feedback step.
   - If no match (`found == False`), the workflow assigns the question to the Answer Writer agent via a fallback task.
3. **Answer Writer Agent (fallback task)**: Uses Azure OpenAI to generate a fallback answer when no FAQ match is found.
4. **Feedback Task**: Presents the answer and captures user feedback.
   - If feedback is negative, the bot triggers a regeneration task with the Answer Writer agent to produce a new LLM answer.

## How It Works

- Users enter a question in the Streamlit UI.
- The bot first tries to answer from the local FAQ database.
- If no FAQ is found, it generates an answer using Azure OpenAI.
- The user can provide feedback; negative feedback triggers a new LLM answer.

## Setup

### Prerequisites

- Python 3.12 or newer
- [pip](https://pip.pypa.io/en/stable/installation/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/) (recommended)

### Python Environment Setup

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

1. Set Azure OpenAI credentials in `.env` (see `.env.example` for template).

### Run the App

```bash
streamlit run app.py
```

## Directory Structure

- `app.py` — main application and workflow
- `faq.db` — SQLite database for FAQs and feedback
- `.env` — Azure OpenAI credentials
- `requirements.txt` — Python dependencies
- `logs/` — application logs

---

This project demonstrates a modular, agentic approach to FAQ automation with human-in-the-loop feedback and LLM fallback using CrewAI.
