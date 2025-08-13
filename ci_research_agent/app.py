# researcher_writer_editor_crewai.py

import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from ddgs import DDGS

# ----------------------- Logging -----------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "researcher_writer_editor.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)

# ----------------------- Env -----------------------
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL_NAME")

# ----------------------- Free Web Search Tool -----------------------


@tool("web_search")
def web_search(query: str, max_results: int = 8, region: str = "us-en") -> str:
    """
    Search the web/news for the given query and return a JSON string:
    {"results":[{"title","url","snippet","date","source"}]}.
    Prefers news results; falls back to text search if needed.
    """
    try:
        results: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            # Try news (more recent); field names vary by version.
            try:
                for r in ddgs.news(query, region=region, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title") or r.get("source") or "",
                            "url": r.get("url") or r.get("link") or "",
                            "snippet": r.get("excerpt") or r.get("body") or "",
                            "date": r.get("date") or r.get("published") or "",
                            "source": r.get("source") or r.get("publisher") or "",
                        }
                    )
            except Exception:
                pass

            if not results:
                # Fallback to general text search
                for r in ddgs.text(query, region=region, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title") or "",
                            "url": r.get("href") or r.get("url") or "",
                            "snippet": r.get("body") or r.get("snippet") or "",
                            "date": r.get("date") or "",
                            "source": r.get("source") or "",
                        }
                    )
        return json.dumps({"results": results})
    except Exception as e:
        logging.exception(e)
        return json.dumps({"results": [], "error": str(e)})


# ----------------------- LLM (Azure via CrewAI) -----------------------
# Keep same pattern as your working CrewAI FAQ agent
crew_llm = LLM(
    model=f"azure/{AZURE_MODEL}",
    api_key=AZURE_API_KEY,
    api_base=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# ----------------------- Agents -----------------------
researcher = Agent(
    role="Researcher",
    goal="Gather recent, credible information on a given topic and summarize concisely with citations.",
    backstory="A pragmatic web researcher who prioritizes recency, source credibility, and concise synthesis.",
    tools=[web_search],
    llm=crew_llm,
    allow_delegation=False,
    verbose=False,
)

writer = Agent(
    role="Writer",
    goal="Draft a short, structured blog post based on the research summary and sources.",
    backstory="A clear, concise technical writer who can turn notes into a readable blog draft.",
    llm=crew_llm,
    allow_delegation=False,
    verbose=False,
)

editor = Agent(
    role="Editor",
    goal="Polish the draft for grammar, tone, and SEO (title, meta, keywords).",
    backstory="A meticulous editor who improves clarity and SEO without changing factual content.",
    llm=crew_llm,
    allow_delegation=False,
    verbose=False,
)

# ----------------------- Tasks -----------------------
# 1) Research: use web_search tool, produce JSON summary + sources
research_task = Task(
    description=(
        "Use the web_search tool to gather recent information on the topic '{topic}'. "
        "Focus on credibility and recency. Produce a JSON object ONLY:\n"
        "{{\n"
        '  "summary": "4-6 bullet points capturing the key insights (no more than ~120 words total)",\n'
        '  "sources": [\n'
        '    {{"title": "...", "url": "https://...", "source": "...", "date": "YYYY-MM-DD"}},\n'
        "    ... up to 3 items MAX\n"
        "  ]\n"
        "}}\n"
        "Rules: Each source MUST have a non-empty http/https URL; skip any without. Limit to 3 sources. Keep JSON compact."
    ),
    expected_output='JSON with keys "summary" (string) and "sources" (list)',
    agent=researcher,
    tools=[web_search],
)

# 2) Write: create a short blog draft from research JSON
write_task = Task(
    description=(
        "You will receive the research JSON from the previous task as context. "
        "Write a concise blog draft (no more than 200 words) about '{topic}'. "
        "Structure in Markdown with sections:\n"
        "## Title (compelling)\n\n"
        "### Introduction\n"
        "### Key Developments\n"
        "### Implications and Outlook\n"
        "### Conclusion\n\n"
        "Embed bracketed citations like [1], [2], [3] referring to the order of 'sources' in the research JSON. "
        "At the end of the post, add a 'References' section listing up to 3 sources as markdown links in the format: [1]: url, [2]: url, [3]: url. "
        "Only include sources with valid, non-empty URLs. Skip any sources with missing or empty URLs. "
        "Do not fabricate facts; rely only on the provided summary and links."
    ),
    expected_output="Markdown blog draft only (with bracketed citations and a References section with up to 3 links).",
    agent=writer,
)

# 3) Edit: polish for grammar/tone + SEO metadata
edit_task = Task(
    description=(
        "Edit the provided draft for grammar, clarity, and tone. Improve SEO by producing:\n"
        "- seo_title (<=60 chars)\n- meta_description (<=160 chars)\n- 5 to 8 keywords\n\n"
        "Return JSON ONLY with keys:\n"
        "{\n"
        '  "final_markdown": "<polished markdown>",\n'
        '  "seo_title": "...",\n'
        '  "meta_description": "...",\n'
        '  "keywords": ["...", "..."]\n'
        "}\n"
        "Keep the content faithful to the sources and citations already included."
    ),
    expected_output='JSON with keys "final_markdown", "seo_title", "meta_description", "keywords".',
    agent=editor,
)

# For apples-to-apples simplicity, we'll run one sequential crew for all three tasks.
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
)

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="🧭 Research → Write → Edit (CrewAI)", layout="wide")
st.title("🧭 Research → Write → Edit (CrewAI + Azure OpenAI)")

if "steps" not in st.session_state:
    st.session_state.steps = []
if "outputs" not in st.session_state:
    st.session_state.outputs = {}

topic = st.text_input(
    "Enter a topic (e.g., AI safety, EV batteries, privacy regulation)"
)
if st.button("Run Pipeline"):
    st.session_state.steps = []
    st.session_state.outputs = {}
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        try:
            result = crew.kickoff(inputs={"topic": topic})
            # Crew returns the final task output (editor JSON). We also reconstruct step logs.
            # For transparency, run a lightweight search to record a 'research step' entry.
            raw_search = web_search.run(topic)
            try:
                parsed_search = json.loads(raw_search)
            except Exception:
                parsed_search = {"results": []}

            # Log steps similar to your FAQ pattern
            st.session_state.steps.append(
                {
                    "step": "research",
                    "result": f"Ran web_search for '{topic}' and summarized sources.",
                }
            )
            st.session_state.steps.append(
                {
                    "step": "write",
                    "result": "Drafted a concise blog post from the research JSON.",
                }
            )
            st.session_state.steps.append(
                {
                    "step": "edit",
                    "result": "Polished the draft for grammar/tone and added SEO metadata.",
                }
            )

            # Parse final editor JSON
            final_obj = {}
            if hasattr(result, "output"):
                # CrewOutput style
                try:
                    final_obj = json.loads(str(result.output))
                except Exception:
                    # If the editor returned plain text for some reason, wrap it
                    final_obj = {"final_markdown": str(result.output)}
            else:
                # String
                try:
                    final_obj = json.loads(str(result))
                except Exception:
                    final_obj = {"final_markdown": str(result)}

            st.session_state.outputs = {
                "raw_search": parsed_search,
                "final": final_obj,
            }
        except Exception as e:
            logging.exception(e)
            st.error(f"Pipeline error: {e}")

# ----------------------- Render -----------------------
if st.session_state.get("outputs"):
    outputs = st.session_state.outputs

    # Research evidence (from tool) — show top 3 hits only
    with st.expander("🔎 Research Sources (from web_search)"):
        hits = outputs.get("raw_search", {}).get("results", [])[:3]
        if not hits:
            st.write("_No sources captured (check duckduckgo-search install)._")
        else:
            for i, h in enumerate(hits, 1):
                title = h.get("title") or "(untitled)"
                url = h.get("url") or ""
                src = h.get("source") or ""
                date = h.get("date") or ""
                snippet = h.get("snippet") or ""
                st.markdown(
                    f"**[{i}] {title}**  \n{snippet}\n\n{src} · {date}  \n{url}"
                )

    # Final polished post + SEO
    final_obj = outputs.get("final", {})
    st.markdown("## ✍️ Final Polished Post")
    final_md = final_obj.get("final_markdown")
    if final_md:
        # Auto-inject a References section (up to 3 valid URLs) if missing
        if "References" not in final_md:
            refs_urls: List[str] = []
            for r in outputs.get("raw_search", {}).get("results", []):
                url = (r.get("url") or "").strip()
                if url.startswith("http"):
                    refs_urls.append(url)
                if len(refs_urls) >= 3:
                    break
            if refs_urls:
                refs_block = "\n".join(f"[{i+1}]: {u}" for i, u in enumerate(refs_urls))
                final_md = (
                    final_md.rstrip() + "\n\n### References\n" + refs_block + "\n"
                )
        st.markdown(final_md)
    else:
        st.info("_No final_markdown returned by editor task._")

    st.markdown("### 🔧 SEO")
    st.write(f"**Title:** {final_obj.get('seo_title', '—')}")
    st.write(f"**Meta:** {final_obj.get('meta_description', '—')}")
    kws = final_obj.get("keywords", []) or []
    if kws:
        st.write("**Keywords:** " + ", ".join(kws))

    # Download
    if final_md:
        st.download_button(
            "⬇️ Download Markdown",
            data=final_md.encode("utf-8"),
            file_name=f"{(topic or 'post').strip().replace(' ','_')}.md",
            mime="text/markdown",
        )
