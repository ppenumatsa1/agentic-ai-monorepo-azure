# LangGraph Math Tutor

A Streamlit-powered math tutor that uses Azure OpenAI and LangGraph to deliver dynamic arithmetic practice, hints, and explanations with adaptive feedback.

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain & LangGraph** (agent workflow)
- **Azure OpenAI** (LLM question generation, grading, hints)
- **Logging** (session logs)

## User Flow

1. **select_item node**: Generates a new arithmetic question using Azure OpenAI, avoiding repeats.
2. **check_answer node**: Grades the user's answer using the LLM.
3. **Branch:**
   - If correct (`is_correct == True`), shows praise and allows the user to proceed.
   - If incorrect, provides a hint (up to 2 attempts), then a worked explanation.
4. **give_hint node**: Offers a short, actionable hint without revealing the answer.
5. **explain node**: Presents a concise, step-by-step solution.
6. **summarize_success node**: Delivers praise and a key takeaway when the answer is correct.

## How It Works

- Users answer dynamically generated arithmetic questions in the Streamlit UI.
- The tutor provides instant grading, hints, and explanations using Azure OpenAI.
- Progress and attempts are tracked; hints and explanations are shown as needed.
- User feedback is collected for each question.

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

1. Set Azure OpenAI credentials in `.env` (see `.env.example` for template).

### Run the App

```bash
streamlit run app.py
```

## Directory Structure

- `app.py` — main application and workflow
- `.env` — Azure OpenAI credentials
- `requirements.txt` — Python dependencies
- `logs/` — session logs

---

This project demonstrates an agentic, adaptive approach to math tutoring with LLM-powered question generation, grading, hints, and explanations.
