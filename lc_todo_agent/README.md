# LangGraph To-Do Agent

A Streamlit-powered to-do list agent that uses Azure OpenAI and LangChain to manage tasks with natural language commands and agentic automation.

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (agent workflow)
- **Azure OpenAI** (LLM-powered task management)
- **SQLite** (task storage)
- **Logging** (session logs)

## User Flow

1. **Add Tasks**: User enters a task description and optional due date in natural language.
2. **List Tasks**: The agent displays all tasks, or only incomplete ones.
3. **Complete Tasks**: User marks tasks as complete by ID or description.
4. **Delete Tasks**: User deletes tasks by ID or description.
5. **Agent Execution**: The agent interprets commands and updates the database.
6. **View Task List**: User sees the current list of tasks and their status.

## How It Works

- The agent uses LLM-powered tools to interpret natural language commands and manage tasks in a SQLite database.
- Supported operations include adding, listing, completing, and deleting tasks.
- All actions are logged for traceability.

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
- `tasks.sqlite` — SQLite database for tasks
- `logs/` — session logs

---

## Example Commands

You can use these commands in the app:

- Add buy milk by tomorrow 9 AM
- Show incomplete tasks
- Complete task #taskid
- Delete task #taskid
- Add "Prepare slides for meeting" due 2025-08-15

---

This project demonstrates an agentic, adaptive approach to task management with LLM-powered automation and natural language commands.
