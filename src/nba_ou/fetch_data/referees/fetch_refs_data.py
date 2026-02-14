import re
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from nba_ou.config.request_headers import HEADERS_BROWSER_LIKE

_ET = ZoneInfo("America/New_York")


def fetch_nba_referee_assignments_today(nba_official_url: str) -> pd.DataFrame:
    """
    Returns the NBA referee assignments table for the current day as posted on official.nba.com.
    Note: the site posts daily assignments (typically same-day only).
    """
    resp = requests.get(
        nba_official_url, headers=HEADERS_BROWSER_LIKE, timeout=30
    )
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    # Pick the NBA table: it has Crew Chief / Referee / Umpire columns.
    nba_tbl = None
    for t in tables:
        cols = set(map(str, t.columns))
        if {"Game", "Crew Chief", "Referee", "Umpire"}.issubset(cols):
            nba_tbl = t.copy()
            break

    if nba_tbl is None:
        raise RuntimeError(
            "Could not locate NBA referee assignments table on the page."
        )

    # Normalize columns (Alternate can be missing sometimes)
    if "Alternate" not in nba_tbl.columns:
        nba_tbl["Alternate"] = pd.NA

    if nba_tbl.empty:
        raise RuntimeError("No referee assignments found for today yet, please check back later.")

    return nba_tbl
