"""
GDELT-Powered Geopolitical Intelligence Briefing System
--------------------------------------------------------
Run: streamlit run app.py
"""

import os
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.agents.intelligence_agent import create_intelligence_agent, generate_brief
from src.ingestion.gdelt_loader import events_to_documents, get_actual_date_range, load_conflict_events
from src.rag.vector_store import build_vector_store, load_vector_store, vector_store_exists

load_dotenv()

st.set_page_config(
    page_title="Geopolitical Intelligence Briefing System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_DATE_RANGE_DAYS = 14  # Hard cap to avoid OOM / health-check timeouts on free-tier hosting

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌐 Intelligence Briefing System")
    st.caption("Powered by GDELT · ChromaDB · Groq LLaMA 3.3 70B")
    st.divider()

    st.subheader("Data Configuration")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.Timestamp("2025-06-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.Timestamp("2025-06-07"))

    selected_days = (end_date - start_date).days
    if selected_days > MAX_DATE_RANGE_DAYS:
        st.warning(
            f"⚠️ Selected range is {selected_days} days. This demo is capped at "
            f"{MAX_DATE_RANGE_DAYS} days to stay within free-tier memory limits. "
            f"The range will be blocked when you click 'Ingest Data & Generate Brief'."
        )

    st.info(
        "📅 **Date note:** GDELT 1.0 files are named by processing date. "
        "Actual event dates shown in the dashboard are typically ~1 year "
        "earlier than the selected file date. All dates are accurate as recorded by GDELT."
    )

    country_options = {
        "Global (all countries)": None,
        "Russia (RS)": "RS",
        "Ukraine (UA)": "UA",
        "Israel (IS)": "IS",
        "Iran (IR)": "IR",
        "China (CH)": "CH",
        "United States (US)": "US",
        "Syria (SY)": "SY",
        "Iraq (IZ)": "IZ",
        "Pakistan (PK)": "PK",
        "Sudan (SU)": "SU",
    }
    selected_label = st.selectbox("Country / Region focus", list(country_options.keys()))
    country_code = country_options[selected_label]

    goldstein_threshold = st.slider(
        "Max Goldstein Scale (conflict severity cutoff)",
        min_value=-10.0,
        max_value=0.0,
        value=-2.0,
        step=0.5,
        help="-10 = maximum hostility, 0 = neutral.",
    )

    st.divider()
    st.subheader("Analysis Focus")
    brief_focus = st.text_input(
        "What should the brief focus on?",
        value=f"{selected_label} conflict dynamics",
    )

    st.divider()
    run_button = st.button(
        "🔍 Ingest Data & Generate Brief",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.caption(f"⚠️ GDELT downloads may take 30-90s per day of data. Max range: {MAX_DATE_RANGE_DAYS} days.")
    if st.button("🗑️ Clear stored vector DB", use_container_width=True):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            st.success("Vector store cleared.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "brief" not in st.session_state:
    st.session_state.brief = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "actual_date_range" not in st.session_state:
    st.session_state.actual_date_range = ""

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
if run_button:
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    if (end_date - start_date).days > MAX_DATE_RANGE_DAYS:
        st.error(
            f"🚫 Date range of {(end_date - start_date).days} days exceeds the "
            f"{MAX_DATE_RANGE_DAYS}-day limit for this demo. This limit exists because "
            f"downloading and embedding more than ~{MAX_DATE_RANGE_DAYS} days of GDELT data "
            f"exceeds the memory available on free-tier hosting and crashes the app. "
            f"Please select a shorter range."
        )
        st.stop()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    date_range_label = f"{start_date.strftime('%b %d')}-{end_date.strftime('%b %d, %Y')}"

    with st.status("📡 Downloading GDELT conflict events...", expanded=True) as status:
        st.write(f"File date range: {date_range_label}")
        st.write(f"Country filter: {selected_label}")
        try:
            df = load_conflict_events(
                start_date=start_str,
                end_date=end_str,
                country_code=country_code,
                goldstein_threshold=goldstein_threshold,
            )
            if df.empty:
                st.error("No conflict events found. Try widening the date range or lowering the Goldstein threshold.")
                st.stop()
            actual_start, actual_end = get_actual_date_range(df)
            st.session_state.df = df
            st.session_state.actual_date_range = f"{actual_start} to {actual_end}"
            status.update(
                label=f"✅ Loaded {len(df):,} conflict events (actual event dates: {actual_start} – {actual_end})",
                state="complete",
            )
        except Exception as e:
            status.update(label="❌ Data download failed", state="error")
            st.error(f"GDELT download error: {e}")
            st.stop()

    with st.status("🧠 Embedding events into vector store...", expanded=True) as status:
        try:
            documents = events_to_documents(df)
            vectorstore = build_vector_store(documents)
            st.session_state.data_loaded = True
            status.update(
                label=f"✅ {len(documents):,} documents embedded into ChromaDB",
                state="complete",
            )
        except Exception as e:
            status.update(label="❌ Embedding failed", state="error")
            st.error(f"Vector store error: {e}")
            st.stop()

    with st.status("🤖 Intelligence agent generating brief...", expanded=True) as status:
        st.write("Agent is querying the vector store and computing metrics...")
        try:
            agent = create_intelligence_agent(vectorstore=vectorstore, df=df)
            actual_range = st.session_state.actual_date_range or date_range_label
            brief = generate_brief(
                agent=agent,
                focus=brief_focus,
                date_range=actual_range,
            )
            st.session_state.brief = brief
            status.update(label="✅ Intelligence brief ready", state="complete")
        except Exception as e:
            status.update(label="❌ Agent failed", state="error")
            st.error(f"Agent error: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    if st.session_state.actual_date_range:
        st.info(f"📅 Actual event dates in this dataset: **{st.session_state.actual_date_range}**")

    tab1, tab2, tab3 = st.tabs(["📊 Conflict Data", "📈 Escalation Analysis", "📋 Intelligence Brief"])

    with tab1:
        st.subheader(f"Conflict Events — {len(df):,} records")
        display_df = df[[
            "SQLDATE", "Actor1Name", "Actor1CountryCode",
            "Actor2Name", "Actor2CountryCode",
            "EventDescription", "GoldsteinScale", "NumArticles",
            "ActionGeo_FullName",
        ]].copy()
        display_df["SQLDATE"] = display_df["SQLDATE"].dt.strftime("%Y-%m-%d")
        display_df["GoldsteinScale"] = display_df["GoldsteinScale"].round(1)
        display_df.columns = [
            "Date", "Actor 1", "A1 Country", "Actor 2", "A2 Country",
            "Event Type", "Goldstein Score", "Articles", "Location",
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Goldstein Score": st.column_config.NumberColumn(
                    format="%.1f",
                    help="-10 = maximum hostility, +10 = maximum cooperation",
                )
            },
        )

    with tab2:
        st.subheader("Escalation Analysis")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Conflict Events", f"{len(df):,}")
        with col_b:
            st.metric("Avg Goldstein Score", f"{df['GoldsteinScale'].mean():.2f}")
        with col_c:
            material = int((df["QuadClass"] == 4).sum())
            st.metric("Material Conflict Events", f"{material:,}")
        with col_d:
            critical = int((df["GoldsteinScale"] <= -7).sum())
            st.metric("Critical Severity Events (≤-7)", f"{critical:,}")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_hist = px.histogram(
                df, x="GoldsteinScale", nbins=30,
                color_discrete_sequence=["#E8593C"],
                title="Goldstein Scale Distribution",
                labels={"GoldsteinScale": "Goldstein Score", "count": "Events"},
            )
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            fig_hist.add_vline(x=df["GoldsteinScale"].mean(), line_dash="dash", line_color="white",
                               annotation_text=f"Mean: {df['GoldsteinScale'].mean():.1f}")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            event_counts = df["EventDescription"].value_counts().head(10)
            fig_bar = px.bar(
                x=event_counts.values, y=event_counts.index, orientation="h",
                color=event_counts.values, color_continuous_scale="Reds_r",
                title="Top 10 Conflict Event Types",
                labels={"x": "Event Count", "y": "Event Type"},
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  showlegend=False, coloraxis_showscale=False,
                                  yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_bar, use_container_width=True)

        daily_counts = (
            df.groupby(df["SQLDATE"].dt.date)
            .agg(events=("GoldsteinScale", "count"), avg_goldstein=("GoldsteinScale", "mean"))
            .reset_index()
        )
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(x=daily_counts["SQLDATE"], y=daily_counts["events"],
                                  name="Event Count", marker_color="#E8593C", opacity=0.7, yaxis="y"))
        fig_time.add_trace(go.Scatter(x=daily_counts["SQLDATE"], y=daily_counts["avg_goldstein"],
                                      name="Avg Goldstein", line=dict(color="#3B8BD4", width=2), yaxis="y2"))
        fig_time.update_layout(
            title="Daily Conflict Events & Average Goldstein Scale",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Event Count"),
            yaxis2=dict(title="Avg Goldstein Score", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with tab3:
        st.subheader("AI-Generated Intelligence Brief")
        if st.session_state.brief:
            st.markdown(st.session_state.brief)
            st.divider()
            st.download_button(
                label="⬇️ Download Brief as .txt",
                data=st.session_state.brief,
                file_name=f"intelligence_brief_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )
        else:
            st.info("No brief generated yet. Click 'Ingest Data & Generate Brief' in the sidebar.")

else:
    st.title("🌐 GDELT Geopolitical Intelligence Briefing System")
    st.markdown(f"""
This system ingests real conflict event data from the **GDELT Project** and uses a
**LangChain agent** powered by **Groq's LLaMA 3.3 70B** to generate structured intelligence briefs.

**How it works:**
1. Select a date range and country focus in the sidebar
2. The system downloads GDELT conflict events (QuadClass 3 & 4)
3. Events are embedded via `sentence-transformers` and stored in **ChromaDB**
4. A **LangGraph tool-calling agent** queries the vector store and computes escalation metrics
5. The agent generates a structured intelligence brief

**Tech stack:** GDELT · Python · LangChain · LangGraph · ChromaDB · Groq · Streamlit

**Note:** This demo is capped at a {MAX_DATE_RANGE_DAYS}-day date range to stay within
the memory limits of free-tier hosting.

---
👈 Configure the analysis in the sidebar and click **Ingest Data & Generate Brief** to begin.
    """)

    with st.expander("📌 GDELT Country Codes Reference"):
        codes = {
            "Russia": "RS", "Ukraine": "UA", "Israel": "IS", "Iran": "IR",
            "China": "CH", "United States": "US", "Syria": "SY", "Iraq": "IZ",
            "Pakistan": "PK", "Sudan": "SU", "Ethiopia": "ET", "Yemen": "YM",
            "Myanmar": "BM", "Afghanistan": "AF", "Libya": "LY",
        }
        code_df = pd.DataFrame(list(codes.items()), columns=["Country", "FIPS Code"])
        st.dataframe(code_df, use_container_width=True, hide_index=True)
