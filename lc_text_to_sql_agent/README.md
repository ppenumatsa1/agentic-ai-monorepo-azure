# LangGraph Text-to-SQL Agent

A Streamlit-powered agent that uses Azure OpenAI and LangChain to answer natural language questions about tabular data by generating and executing SQL queries.

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (agent workflow)
- **Azure OpenAI** (LLM-powered SQL generation)
- **Logging** (session logs)

## User Flow

1. **Upload Data**: User uploads a CSV file (single table).
2. **Preview Schema**: The app displays the table schema.
3. **Ask Questions**: User enters natural language questions about the data.
4. **Agent Execution**: The agent generates SQL queries, executes them, and returns natural language answers.
5. **View Results**: User sees the answer and can inspect intermediate steps.

## How It Works

- The agent uses LLM-powered tools to interpret questions, generate SQL, and answer in natural language.
- Only safe SELECT queries are allowed (max 50 rows).
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
- `data/` — sample CSV files
- `logs/` — session logs

---

## Example Questions & Expected Answers

You can use these questions in the app with the provided sample data:

### Users Table (`users.csv`)

- How many users signed up in March 2025?
  - Expected answer: There are 2 users who signed up in March 2025.
- List user IDs and names of people whose name starts with ‘A’ or ‘B’.
  - Expected answer: The user IDs and names of people whose names start with 'A' or 'B' are: user ID 1 with the name Alice, and user ID 2 with the name Bob.
- When did user_id = 4 sign up?
  - Expected answer: User ID 4 signed up on 2025-03-01.
- Show all users who signed up before February 2025.
  - Expected answer: There is one user who signed up before February 2025: Alice, with the email alice@example.com, who signed up on January 15, 2025.
- Count the total number of users in the database.
  - Expected answer: The total number of users in the database is 5.

### Orders Table (`orders.csv`)

- How many orders were placed in March 2025?
  - Expected answer: There were 4 orders placed in March 2025.
- What is the total order amount for each user_id?
  - Expected answer: The total order amount for user_id 1 is 325.75, for user_id 2 is 300.0, and for user_id 3 is 125.0.
- Show order_id, user_id, and amount for orders over 100.
  - Expected answer: There are three orders with amounts over 100: order ID 101 by user ID 1 with an amount of 250.5, order ID 104 by user ID 3 with an amount of 125.0, and order ID 105 by user ID 2 with an amount of 200.0.
- What is the average order amount across all orders?
  - Expected answer: The average order amount across all orders is 150.15.
- List all orders placed by user_id = 2.
  - Expected answer: User with user_id 2 placed two orders: order_id 102 on 2025-03-07 with an amount of 100.0, and order_id 105 on 2025-04-01 with an amount of 200.0.

---

This project demonstrates an agentic, adaptive approach to tabular data Q&A with LLM-powered SQL generation and natural language answers.
