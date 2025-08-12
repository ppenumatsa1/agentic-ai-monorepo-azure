# LangGraph Data Wrangler Agent

A Streamlit-powered data cleaning agent that uses Azure OpenAI and LangChain to automate and guide data wrangling tasks for messy tabular data.

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (agent workflow)
- **Azure OpenAI** (LLM-powered cleaning instructions)
- **Logging** (session logs)

## User Flow

1. **Upload Data**: User uploads a CSV or XLSX file.
2. **Preview Data**: The app displays the raw data for review.
3. **Enter Cleaning Instructions**: User provides natural language instructions (see examples below).
4. **Agent Execution**: The agent interprets instructions, applies cleaning tools, and updates the data.
5. **Download Cleaned Data**: User can download the cleaned CSV.
6. **Reset Data**: Option to revert to the original upload.

## How It Works

- The agent uses LLM-powered tools to automate common data cleaning tasks.
- Supported operations include removing duplicates, standardizing dates, filling missing values, extracting emails, and dropping columns with excessive missing data.
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
- `logs/` — session logs

---

## Example Cleaning Instructions

You can use these instructions in the app:

- Remove duplicates
- Standardize dates in column "date" to format YYYY-MM-DD
- Fill missing values in column "email" with constant value "unknown@example.com"
- Extract emails from column "email" to new column "extracted_email"
- Drop columns with missing fraction above 0.5
- Fill missing values in column "comments" with constant value "No comment"

---

This project demonstrates an agentic, adaptive approach to data wrangling with LLM-powered cleaning tools and natural language instructions.
