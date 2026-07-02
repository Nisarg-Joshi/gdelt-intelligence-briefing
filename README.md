# GDELT Geopolitical Intelligence Briefing System

An agentic RAG pipeline that ingests real-world conflict event data from the [GDELT Project](https://www.gdeltproject.org/) and generates structured, plain-language intelligence briefs using a LangGraph tool-calling agent powered by Groq's LLaMA 3.3 70B.

**Live demo:** [Launch Live Demo](https://gdelt-intelligence-briefing-hrydr8rrgmqynk2waeedzr.streamlit.app/)

---

## What it does

1. **Ingests** real conflict event data directly from GDELT 1.0's public archive — no synthetic or hardcoded data
2. **Filters** events by conflict type (CAMEO QuadClass 3/4: verbal and material conflict) and hostility severity (Goldstein Scale)
3. **Embeds** every event into a local vector store using `sentence-transformers` and ChromaDB — fully CPU-based, no GPU required
4. **Reasons** over the data using a LangGraph agent with three tools: retrieving the most severe events, semantic search over the event corpus, and computing country-level escalation statistics
5. **Generates** a structured intelligence brief in plain English — not a database dump, but a readable analysis naming specific actors, locations, and events

## Why this project

Most portfolio RAG projects use synthetic or toy datasets. This one is built on a real open-source conflict-monitoring dataset used by actual researchers and risk analysts, and demonstrates an end-to-end agentic system: data ingestion, vector retrieval, tool-calling, and grounded text generation — not just an LLM wrapper.

## Architecture

```
GDELT CSV (raw events)
        │
        ▼
gdelt_loader.py  — filter for conflict events, compute severity
        │
        ▼
vector_store.py  — embed via MiniLM, store in ChromaDB
        │
        ▼
intelligence_agent.py  — LangGraph agent with 3 tools:
   • get_top_severe_events   (most hostile events by Goldstein score)
   • search_conflict_events  (semantic search over the corpus)
   • get_escalation_metrics  (country-level statistics)
        │
        ▼
app.py  — Streamlit dashboard: data table, charts, generated brief
```

## Tech stack

- **Data source:** GDELT Project (1.0 event database)
- **Orchestration:** LangChain / LangGraph
- **LLM:** Groq — LLaMA 3.3 70B
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (local, persistent)
- **Frontend:** Streamlit + Plotly
- **Language:** Python

## A known data quirk — and how it's handled

GDELT 1.0 files are named by their processing date, but the events inside a file are typically dated about a year earlier than the filename. Rather than silently mismatching dates, this project surfaces the **actual event date range** directly from the data in the dashboard, with a clear note explaining why. Transparency about data limitations was a deliberate design choice here, not an oversight.

## Running it locally

```bash
git clone https://github.com/Nisarg-Joshi/gdelt-intelligence-briefing.git
cd gdelt-intelligence-briefing
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

Then run:

```bash
streamlit run app.py
```

## Project structure

```
gdelt-intelligence-briefing/
├── app.py                          # Streamlit dashboard (entry point)
├── requirements.txt
├── src/
│   ├── ingestion/
│   │   └── gdelt_loader.py         # GDELT download + filtering
│   ├── rag/
│   │   └── vector_store.py         # Embeddings + ChromaDB
│   └── agents/
│       └── intelligence_agent.py   # LangGraph agent + tools
```

## Author

**Nisarg Joshi** — AI/ML Engineer
[GitHub](https://github.com/Nisarg-Joshi) · [LinkedIn](https://linkedin.com/in/nisarg-joshi-528866222) · [Portfolio](https://nisargjoshi282.netlify.app)
