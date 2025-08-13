# Agentic AI Monorepo: The Art of Possible

This monorepo showcases the art of possible with modern agent frameworks, including LangChain, LangGraph, CrewAI, Azure AI Foundry Agents, Semantic Kernel, and more. Each project demonstrates a unique approach to building intelligent and adaptive agents for real-world tasks.

## Repository Structure & Naming Convention

- Folders prefixed with `lc_` use **LangChain**
- Folders prefixed with `lg_` use **LangGraph**
- Folders prefixed with `ci_` use **CrewAI**
- Folders prefixed with `sk_` use **Semantic Kernel**
- Other folders may use Azure AI Foundry or custom agent frameworks

## Projects in This Monorepo

| Project Folder                                        | Description                                         | README                                       |
| ----------------------------------------------------- | --------------------------------------------------- | -------------------------------------------- |
| [`lc_data_wrangler_agent`](./lc_data_wrangler_agent/) | LLM-powered data cleaning agent (LangChain)         | [README](./lc_data_wrangler_agent/README.md) |
| [`lc_text_to_sql_agent`](./lc_text_to_sql_agent/)     | Text-to-SQL agent for tabular data (LangChain)      | [README](./lc_text_to_sql_agent/README.md)   |
| [`lc_todo_agent`](./lc_todo_agent/)                   | AI-powered to-do list agent (LangChain)             | [README](./lc_todo_agent/README.md)          |
| [`lg_faq_agent`](./lg_faq_agent/)                     | FAQ bot with local DB and LLM fallback (LangGraph)  | [README](./lg_faq_agent/README.md)           |
| [`lg_tutor_agent`](./lg_tutor_agent/)                 | Math tutor agent with adaptive feedback (LangGraph) | [README](./lg_tutor_agent/README.md)         |
| [`ci_faq_agent`](./ci_faq_agent/)                     | FAQ bot with local DB and LLM fallback (CrewAI)     | [README](./ci_faq_agent/README.md)           |
| [`ci_research_agent`](./ci_research_agent/)           | Research and blog-writing agent (CrewAI)            | [README](./ci_research_agent/README.md)      |
| ...                                                   | More agent demos coming soon                        |                                              |

## General Overview

This repository is designed to:

- Demonstrate art of possible in agentic AI development
- Compare and contrast different agent frameworks and orchestration strategies
- Serve as a starting point for building modular, scalable, and human-in-the-loop AI agents
- Inspire experimentation and rapid prototyping with LLMs and agent workflows

Each project includes its own README with setup instructions, tech stack, user flow, and example usage. Explore the subdirectories to learn more about each agent and framework.

---

**Explore, experiment, and push the boundaries of agentic AI!**
