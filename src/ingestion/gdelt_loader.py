"""
GDELT Ingestion Module
----------------------
Downloads GDELT 1.0 daily export files, filters for conflict events,
and converts rows into natural language text documents for embedding.

GDELT 1.0 note:
Files are named by the date they were processed. The SQLDATE field inside
each file reflects the actual date the event occurred, which is typically
around one year prior to the filename date. All dates displayed in the
dashboard are taken directly from SQLDATE — they are always accurate.

QuadClass 3 = Verbal Conflict, 4 = Material Conflict
GoldsteinScale: -10 (most hostile) to +10 (most cooperative)
"""

import io
import logging
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GDELT_BASE_URL = "http://data.gdeltproject.org/events/"

GDELT_COLUMNS = [
    "GlobalEventID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

CAMEO_ROOT_DESCRIPTIONS = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit military posture",
    "16": "Reduce diplomatic relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use unconventional mass violence",
}


def _describe_event(event_code: str) -> str:
    root = str(event_code)[:2] if event_code else "??"
    return CAMEO_ROOT_DESCRIPTIONS.get(root, f"Event type {event_code}")


def download_gdelt_day(date: datetime) -> pd.DataFrame:
    """
    Download and parse a single day of GDELT 1.0 data.
    The filename date is the processing date. SQLDATE inside reflects
    actual event dates, typically ~1 year prior to filename.
    """
    date_str = date.strftime("%Y%m%d")
    url = f"{GDELT_BASE_URL}{date_str}.export.CSV.zip"
    logger.info(f"Downloading GDELT data for {date_str}...")

    response = requests.get(url, timeout=90)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS,
                low_memory=False,
                on_bad_lines="skip",
            )

    logger.info(f"  -> {len(df):,} raw events loaded for {date_str}")
    return df


def get_actual_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """Return the actual min/max SQLDATE from the loaded data as strings."""
    if df.empty or "SQLDATE" not in df.columns:
        return ("unknown", "unknown")
    min_date = df["SQLDATE"].min().strftime("%B %d, %Y")
    max_date = df["SQLDATE"].max().strftime("%B %d, %Y")
    return (min_date, max_date)


def load_conflict_events(
    start_date: str,
    end_date: str,
    country_code: str | None = None,
    goldstein_threshold: float = -2.0,
) -> pd.DataFrame:
    """
    Load and filter conflict events across a date range.

    Parameters
    ----------
    start_date : str  - "YYYY-MM-DD" (GDELT file date, not event date)
    end_date   : str  - "YYYY-MM-DD"
    country_code : str | None
    goldstein_threshold : float
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if (end - start).days > 14:
        logger.warning("Date range exceeds 14 days. Consider a shorter window for demos.")

    all_dfs = []
    current = start
    while current <= end:
        try:
            df = download_gdelt_day(current)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load data for {current.date()}: {e}")
        current += timedelta(days=1)

    if not all_dfs:
        logger.error("No GDELT data could be downloaded.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Parse SQLDATE - reflects actual event dates
    combined["SQLDATE"] = pd.to_datetime(
        combined["SQLDATE"], format="%Y%m%d", errors="coerce"
    )

    # Filter: conflict events only
    conflict = combined[combined["QuadClass"].isin([3, 4])].copy()

    # Filter: minimum conflict severity
    conflict["GoldsteinScale"] = pd.to_numeric(
        conflict["GoldsteinScale"], errors="coerce"
    )
    conflict = conflict[conflict["GoldsteinScale"] <= goldstein_threshold]

    # Filter: country
    if country_code:
        code = country_code.upper()
        conflict = conflict[
            (conflict["Actor1CountryCode"] == code)
            | (conflict["Actor2CountryCode"] == code)
            | (conflict["ActionGeo_CountryCode"] == code)
        ]

    # Select key columns
    key_cols = [
        "SQLDATE", "Actor1Name", "Actor1CountryCode",
        "Actor2Name", "Actor2CountryCode",
        "EventCode", "EventRootCode", "QuadClass",
        "GoldsteinScale", "NumArticles", "AvgTone",
        "ActionGeo_FullName", "ActionGeo_CountryCode",
        "ActionGeo_Lat", "ActionGeo_Long",
        "SOURCEURL",
    ]
    conflict = conflict[key_cols].copy()

    for col in ["GoldsteinScale", "NumArticles", "AvgTone",
                "ActionGeo_Lat", "ActionGeo_Long"]:
        conflict[col] = pd.to_numeric(conflict[col], errors="coerce")

    conflict["EventDescription"] = conflict["EventRootCode"].astype(str).apply(
        _describe_event
    )

    conflict = conflict.dropna(subset=["GoldsteinScale", "SQLDATE"])
    conflict = conflict.sort_values("SQLDATE").reset_index(drop=True)

    logger.info(
        f"Final conflict dataset: {len(conflict):,} events "
        f"({'global' if not country_code else country_code})"
    )
    return conflict


def events_to_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each DataFrame row into a text document + metadata dict
    suitable for embedding and storing in ChromaDB.
    """
    documents = []

    for _, row in df.iterrows():
        date_str = (
            row["SQLDATE"].strftime("%B %d, %Y")
            if pd.notnull(row["SQLDATE"])
            else "Unknown date"
        )
        actor1 = row["Actor1Name"] if pd.notnull(row["Actor1Name"]) else "Unknown party"
        actor2 = row["Actor2Name"] if pd.notnull(row["Actor2Name"]) else "Unknown party"
        a1_country = row["Actor1CountryCode"] if pd.notnull(row["Actor1CountryCode"]) else "?"
        a2_country = row["Actor2CountryCode"] if pd.notnull(row["Actor2CountryCode"]) else "?"
        location = (
            row["ActionGeo_FullName"]
            if pd.notnull(row["ActionGeo_FullName"])
            else "Unknown location"
        )
        action_country = (
            row["ActionGeo_CountryCode"]
            if pd.notnull(row["ActionGeo_CountryCode"])
            else "?"
        )
        goldstein = float(row["GoldsteinScale"])
        event_desc = row.get("EventDescription", "conflict event")
        articles = int(row["NumArticles"]) if pd.notnull(row["NumArticles"]) else 0

        if goldstein <= -7:
            severity = "CRITICAL severity"
            severity_plain = "an extremely hostile incident"
        elif goldstein <= -5:
            severity = "HIGH severity"
            severity_plain = "a highly hostile incident"
        elif goldstein <= -3:
            severity = "MEDIUM severity"
            severity_plain = "a moderately hostile incident"
        else:
            severity = "LOW severity"
            severity_plain = "a low-level hostile incident"

        if actor1 != "Unknown party" and actor2 != "Unknown party":
            verb_phrase = f"{actor1} carried out a {event_desc.lower()} action against {actor2}"
        elif actor1 != "Unknown party":
            verb_phrase = f"{actor1} carried out a {event_desc.lower()} action"
        elif actor2 != "Unknown party":
            verb_phrase = f"A {event_desc.lower()} action was carried out against {actor2}"
        else:
            verb_phrase = f"A {event_desc.lower()} incident took place"

        if articles >= 10:
            coverage = f"widely covered by {articles} media outlets"
        elif articles >= 3:
            coverage = f"reported by {articles} media outlets"
        else:
            coverage = f"noted in {articles} media report(s)"

        text = (
            f"On {date_str}, {verb_phrase} in {location}. "
            f"This was {severity_plain} with a hostility score of {goldstein:.1f} "
            f"on a scale where -10 means maximum warfare and 0 means neutral. "
            f"The event was {coverage}. "
            f"{'This was a physical/military action.' if row['QuadClass'] == 4 else 'This was a verbal or diplomatic conflict action.'}"
        )

        metadata = {
            "date": date_str,
            "action_country": str(action_country),
            "actor1_country": str(a1_country),
            "actor2_country": str(a2_country),
            "goldstein": goldstein,
            "quad_class": int(row["QuadClass"]),
            "num_articles": articles,
            "severity": severity,
            "location": str(location),
        }

        documents.append({"text": text, "metadata": metadata})

    return documents