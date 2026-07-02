"""
Intelligence Agent Module
-------------------------
A LangGraph tool-calling agent powered by Groq (LLaMA 3.3 70B) that
analyses geopolitical conflict events and generates structured intelligence briefs.

Agent tools:
  1. search_conflict_events  — semantic RAG retrieval from ChromaDB
  2. get_escalation_metrics  — statistical Goldstein scale analysis from DataFrame

Architecture: langgraph.prebuilt.create_react_agent (replaces deprecated AgentExecutor)
"""

import json
import logging
import os

import pandas as pd
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a geopolitical intelligence analyst writing for a general audience \
— educated but not specialists. Your job is to explain what is actually happening in a \
conflict zone in plain, specific, human language. No jargon. No vague summaries.

When asked to analyse a geopolitical situation, you MUST call your tools in this order:
1. Call get_top_severe_events FIRST — this gives you the most dramatic events in the data
2. Call search_conflict_events for military/fighting events
3. Call search_conflict_events again for diplomatic/political events
4. Call get_escalation_metrics for the relevant country codes
5. Write the brief using ONLY what you found in steps 1-4

CRITICAL WRITING RULES — follow these without exception:

RULE 1 — BE SPECIFIC, NEVER GENERIC
Bad: "There were multiple material conflict events of HIGH severity involving various parties."
Good: "Ukrainian forces conducted assault operations near Kursk, Russia's western border \
region, with 16 separate media reports confirming the activity on June 1 alone."

RULE 2 — NAME REAL THINGS
Always name the specific cities, regions, organizations, and people from the retrieved data. \
Never write "an actor" or "a location" or "various parties". \
If the data says Kursk — write Kursk. If it says Vladimir Putin — write Vladimir Putin. \
If it says the UN rejected a Russian proposal in Moscow — say exactly that.

RULE 3 — TRANSLATE NUMBERS INTO MEANING
Bad: "The average Goldstein scale score is -6.74."
Good: "On a hostility scale where -10 means all-out warfare and 0 means neutral, \
the average score across these events was -6.7. That places the Russia situation \
firmly in active conflict territory — far closer to open war than to mere diplomatic tension."

RULE 4 — EXPLAIN WHAT EVENTS ACTUALLY MEAN IN PRACTICE
Bad: "There were 2,424 material conflict events."
Good: "More than half of all recorded incidents involved physical actions — actual fighting, \
assaults, and coercive force — rather than threats or diplomatic statements. \
When material events outnumber verbal ones, it signals the conflict has moved beyond rhetoric."

RULE 5 — EVERY SENTENCE MUST ANSWER 'SO WHAT?'
If a sentence doesn't tell the reader something concrete and meaningful, rewrite it. \
The reader should finish each section knowing something specific they didn't know before.

Output your intelligence brief in this exact structure:

---
**INTELLIGENCE BRIEF**
**Date of Analysis:** [today's date]
**Region / Country of Focus:** [as specified]
**Classification:** UNCLASSIFIED // FOR DEMONSTRATION PURPOSES

**1. WHAT IS HAPPENING RIGHT NOW**
[2-3 paragraphs. Describe the actual situation on the ground in plain English. \
Name specific cities, regions, actors, and events pulled from your search results. \
A reader should finish this section knowing exactly what is going on and where — \
not a summary of event categories, but a description of actual events.]

**2. WHO IS INVOLVED AND WHAT ARE THEY DOING**
[Bullet list. Name each actor specifically and describe their concrete actions \
— not their abstract "role" but what they are physically or diplomatically doing \
based on the retrieved events.]

**3. HOW SERIOUS IS THIS**
**Risk Level: [LOW / MEDIUM / HIGH / CRITICAL]**
[Explain the risk level in plain English first, then use the metrics to back it up \
with translated meaning. Tell the reader whether this is getting worse or better \
and what specifically indicates that direction.]

**4. WHAT THE PATTERN OF EVENTS TELLS US**
[Look at the types of events retrieved. Are they mostly fighting, threats, \
diplomatic breakdowns, or a mix? Explain what that pattern means for where \
this conflict is heading. Be concrete — not "the situation is complex" \
but "the dominance of X type of event suggests Y specific trajectory."]

**5. WHY THIS MATTERS BEYOND THE REGION**
[2-3 sentences in plain terms. Who else is affected, how, and why should \
someone outside the region care. Name specific countries, institutions, or \
economic interests where the data supports it.]

**6. WHAT TO WATCH FOR NEXT**
[3-5 bullet points. Specific observable signals — not vague categories. \
Name specific borders, institutions, leaders, or military assets whose \
movement or statement would indicate escalation or de-escalation.]
---"""


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def _make_tools(vectorstore: Chroma, df: pd.DataFrame):
    """
    Create the two agent tools as closures so they have access to
    the vectorstore and dataframe without global state.
    """

    @tool
    def get_top_severe_events(n: int = 5) -> str:
        """
        Retrieve the most severe conflict events by hostility score.
        Always call this FIRST — it gives you the most dramatic and
        newsworthy events in the dataset, sorted by hostility.
        Returns the top N events sorted by Goldstein scale (most hostile first).
        """
        if df.empty:
            return "No data loaded."
        top = df.nsmallest(n, "GoldsteinScale")
        lines = []
        for _, row in top.iterrows():
            date_str = (
                row["SQLDATE"].strftime("%B %d, %Y")
                if pd.notnull(row["SQLDATE"])
                else "Unknown date"
            )
            actor1 = row["Actor1Name"] if pd.notnull(row["Actor1Name"]) else "Unknown"
            actor2 = row["Actor2Name"] if pd.notnull(row["Actor2Name"]) else "Unknown"
            location = (
                row["ActionGeo_FullName"]
                if pd.notnull(row["ActionGeo_FullName"])
                else "Unknown location"
            )
            goldstein = float(row["GoldsteinScale"])
            event_desc = row.get("EventDescription", "conflict")
            articles = int(row["NumArticles"]) if pd.notnull(row["NumArticles"]) else 0
            lines.append(
                f"- {date_str} | {event_desc} | {actor1} vs {actor2} | "
                f"Location: {location} | Hostility score: {goldstein:.1f} out of -10 | "
                f"Media articles: {articles}"
            )
        return "\n".join(lines)

    @tool
    def search_conflict_events(query: str) -> str:
        """
        Semantic search over the GDELT conflict event database.
        Use descriptive queries to find relevant events, e.g.:
          - 'military clashes and fighting'
          - 'diplomatic breakdown and rejection'
          - 'military posture and threats'
        Returns the top 3 most relevant event descriptions.
        """
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "No relevant conflict events found for this query."
        formatted = []
        for i, doc in enumerate(results):
            content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
            formatted.append(f"[Event {i + 1}]\n{content}")
        return "\n\n".join(formatted)

    @tool
    def get_escalation_metrics(country_code: str) -> str:
        """
        Compute statistical escalation metrics for a given country.
        Provide a two-letter FIPS country code, e.g.:
          RS = Russia, UA = Ukraine, IZ = Iraq, SY = Syria,
          CH = China, US = United States, IR = Iran, IS = Israel
        Returns Goldstein scale statistics and event counts as JSON.
        """
        if df.empty:
            return json.dumps({"error": "No data loaded."})

        code = country_code.strip().upper()
        mask = (
            (df["Actor1CountryCode"] == code)
            | (df["Actor2CountryCode"] == code)
            | (df["ActionGeo_CountryCode"] == code)
        )
        country_df = df[mask]

        if country_df.empty:
            return json.dumps(
                {
                    "country_code": code,
                    "note": f"No conflict events found for country code '{code}' "
                    "in the loaded dataset. Try a different code.",
                }
            )

        verbal = int((country_df["QuadClass"] == 3).sum())
        material = int((country_df["QuadClass"] == 4).sum())
        avg_g = float(country_df["GoldsteinScale"].mean())
        min_g = float(country_df["GoldsteinScale"].min())
        max_g = float(country_df["GoldsteinScale"].max())
        severe = int((country_df["GoldsteinScale"] <= -7).sum())
        high = int(
            ((country_df["GoldsteinScale"] > -7) & (country_df["GoldsteinScale"] <= -5)).sum()
        )
        medium = int(
            ((country_df["GoldsteinScale"] > -5) & (country_df["GoldsteinScale"] <= -3)).sum()
        )
        total_articles = int(country_df["NumArticles"].sum())

        if avg_g <= -7:
            escalation_level = "CRITICAL"
        elif avg_g <= -5:
            escalation_level = "HIGH"
        elif avg_g <= -3:
            escalation_level = "MEDIUM"
        else:
            escalation_level = "LOW"

        metrics = {
            "country_code": code,
            "total_conflict_events": verbal + material,
            "verbal_conflict_events": verbal,
            "material_conflict_events": material,
            "goldstein_scale": {
                "average": round(avg_g, 2),
                "minimum_most_hostile": round(min_g, 2),
                "maximum_least_hostile": round(max_g, 2),
                "note": "Scale runs from -10 (maximum hostility) to +10 (maximum cooperation)",
            },
            "severity_breakdown": {
                "CRITICAL_score_lte_neg7": severe,
                "HIGH_score_lte_neg5": high,
                "MEDIUM_score_lte_neg3": medium,
            },
            "total_media_articles_covering_events": total_articles,
            "derived_escalation_level": escalation_level,
        }
        return json.dumps(metrics, indent=2)

    return [get_top_severe_events, search_conflict_events, get_escalation_metrics]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def create_intelligence_agent(
    vectorstore: Chroma,
    df: pd.DataFrame,
    groq_api_key: str | None = None,
):
    """
    Build and return the LangGraph react agent.

    Parameters
    ----------
    vectorstore : Chroma — the loaded vector store
    df : pd.DataFrame   — the raw conflict events DataFrame
    groq_api_key : str  — if None, reads from GROQ_API_KEY env var

    Returns
    -------
    LangGraph agent app ready to invoke with {"messages": [...]}
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file or environment."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=api_key,
        max_tokens=2048,
    )

    tools = _make_tools(vectorstore, df)
    agent = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
    logger.info("Intelligence agent ready.")
    return agent


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def generate_brief(
    agent,
    focus: str,
    date_range: str,
) -> str:
    """
    Run the agent and return the intelligence brief as a string.

    Parameters
    ----------
    agent      — the LangGraph agent returned by create_intelligence_agent
    focus      : str — e.g. "Russia-Ukraine conflict" or "Middle East"
    date_range : str — e.g. "June 1-7, 2025"

    Returns
    -------
    str — the formatted intelligence brief
    """
    query = (
        f"Generate a comprehensive intelligence brief for the following:\n\n"
        f"Focus area: {focus}\n"
        f"Date range covered by the data: {date_range}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. First search for specific military and fighting events related to {focus}\n"
        f"2. Then search for diplomatic and political events related to {focus}\n"
        f"3. Then get escalation metrics for the relevant country code\n"
        f"4. Write the brief using SPECIFIC named locations, actors, and events \n"
        f"   from your search results — not generic summaries.\n"
        f"5. Every claim must reference a specific event from the data.\n"
        f"6. Translate all technical scores into plain English meaning."
    )
    logger.info(f"Running intelligence agent for: {focus} | {date_range}")
    result = agent.invoke({"messages": [("human", query)]})
    # LangGraph returns messages list — last message is the final answer
    return result["messages"][-1].content