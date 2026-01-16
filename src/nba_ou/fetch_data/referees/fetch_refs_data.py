import re
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import requests

NBA_OFFICIAL_ASSIGNMENTS_URL = "https://official.nba.com/referee-assignments/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


_ET = ZoneInfo("America/New_York")


def _norm_matchup(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("los angeles", "la")
    s = re.sub(r"[^a-z0-9@ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_nba_referee_assignments_today() -> pd.DataFrame:
    """
    Returns the NBA referee assignments table for the current day as posted on official.nba.com.
    Note: the site posts daily assignments (typically same-day only).
    """
    resp = requests.get(
        NBA_OFFICIAL_ASSIGNMENTS_URL, headers=DEFAULT_HEADERS, timeout=30
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

    nba_tbl["MATCHUP_KEY"] = nba_tbl["Game"].map(lambda x: _norm_matchup(str(x)))
    return nba_tbl
