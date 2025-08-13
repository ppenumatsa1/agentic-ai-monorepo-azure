# CrewAI Research Blog Agent

A Streamlit-powered research and blog-writing agent that combines web search with Azure OpenAI, built using CrewAI and DDGS (DuckDuckGo Search).

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **CrewAI** (agent workflow)
- **Azure OpenAI** (LLM)
- **DDGS** (web search)

## User Flow

1. **Researcher Agent (with web_search tool and research task)**: Receives the user's topic and uses the `web_search` tool to gather recent, credible information from the web (via DDGS).
2. **Writer Agent (write task)**: Takes the research summary and sources, then drafts a concise, structured blog post (Markdown, <200 words) with bracketed citations.
3. **Editor Agent (edit task)**: Polishes the draft for grammar, tone, and SEO, returning the final Markdown, SEO title, meta description, and keywords.
4. **Streamlit UI**: Users enter a topic, run the pipeline, view the research sources, final post, SEO metadata, and can download the Markdown.

## How It Works

- Users enter a topic in the Streamlit UI.
- The agent pipeline runs: research → write → edit.
- The app displays the top 3 research sources, the final polished post (with citations and references), and SEO info.
- Users can download the final Markdown post.

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
2. (Optional) To disable CrewAI telemetry, add to `.env`:
   ```
   CREWAI_DISABLE_TELEMETRY=1
   ```

### Run the App

```bash
streamlit run app.py
```

## Directory Structure

- `app.py` — main application and workflow
- `.env` — Azure OpenAI credentials and config
- `requirements.txt` — Python dependencies
- `logs/` — application logs

---

This project demonstrates a modular, agentic approach to research and blog automation with web search, LLM summarization, and SEO optimization using CrewAI.
