"""Fetch authoritative NBA tipoff times from the public season-schedule feed.

One request returns a whole season, which makes this the cheap way to get
``tipoff_utc`` for every ``game_id`` -- the per-game ``cdn.nba.com`` boxscore
endpoint used by ``game_time_index`` needs one request per game instead.

The feed exposes both an Eastern wall-clock time (``etm``) and a UTC date/time
pair (``gdtutc`` / ``utctm``); we prefer the UTC pair and fall back to
localizing the Eastern value.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

SCHEDULE_URL = (
    "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/"
    "{season_year}/league/00_full_schedule.json"
)

# data.nba.com returns 403 without a browser-like Referer/Origin pair.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}

EASTERN_TZ = ZoneInfo("America/New_York")

SCHEDULE_COLUMNS = [
    "game_id",
    "season_year",
    "game_date",
    "tipoff_utc",
    "tipoff_et",
    "team_home",
    "team_away",
    "arena_name",
    "arena_city",
    "game_status",
]


def _parse_tipoff_utc(game: dict) -> pd.Timestamp | None:
    """Prefer the explicit UTC pair; fall back to localizing Eastern time."""
    utc_date = (game.get("gdtutc") or "").strip()
    utc_time = (game.get("utctm") or "").strip()
    if utc_date and utc_time:
        parsed = pd.to_datetime(f"{utc_date} {utc_time}", errors="coerce", utc=True)
        if not pd.isna(parsed):
            return parsed

    eastern = (game.get("etm") or "").strip()
    if eastern:
        try:
            naive = datetime.fromisoformat(eastern)
        except ValueError:
            return None
        return pd.Timestamp(naive.replace(tzinfo=EASTERN_TZ)).tz_convert("UTC")

    return None


def fetch_season_schedule(season_year: int, *, timeout: int = 30) -> pd.DataFrame:
    """Return one row per game for ``season_year`` (2024 == the 2024-25 season)."""
    response = requests.get(
        SCHEDULE_URL.format(season_year=season_year),
        headers=_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()

    rows: list[dict] = []
    for month in response.json().get("lscd", []):
        for game in month.get("mscd", {}).get("g", []):
            game_id = (game.get("gid") or "").strip()
            if not game_id:
                continue

            eastern = (game.get("etm") or "").strip()
            rows.append(
                {
                    "game_id": game_id,
                    "season_year": season_year,
                    "game_date": pd.to_datetime(
                        game.get("gdte"), errors="coerce"
                    ).date(),
                    "tipoff_utc": _parse_tipoff_utc(game),
                    "tipoff_et": eastern or None,
                    "team_home": (game.get("h") or {}).get("ta"),
                    "team_away": (game.get("v") or {}).get("ta"),
                    "arena_name": game.get("an"),
                    "arena_city": game.get("ac"),
                    "game_status": game.get("st"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    return pd.DataFrame(rows)[SCHEDULE_COLUMNS]


BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"


def fetch_tipoffs_for_dates(
    game_dates: list, *, timeout: int = 90, season_year: int | None = None
) -> pd.DataFrame:
    """Tipoff fallback, by date, for games missing from the season feed.

    The static season feed is published before the season and never backfilled,
    so it has a hole wherever fixtures are scheduled later -- notably the NBA
    Cup window, where the non-qualifying teams' makeup games are added once the
    bracket is known. The daily scoreboard is authoritative for those.

    ``ScoreboardV3`` rather than V2: V2 has known line-score defects over
    2025-10-22..2025-12-25, which is exactly the affected window.
    """
    from nba_api.stats.endpoints import ScoreboardV3

    rows: list[dict] = []
    for game_date in sorted({str(d) for d in game_dates}):
        try:
            payload = ScoreboardV3(game_date=game_date, timeout=timeout).get_dict()
        except Exception:  # noqa: BLE001 - a bad date must not abort the load
            continue

        for game in payload.get("scoreboard", {}).get("games", []):
            tipoff = pd.to_datetime(game.get("gameTimeUTC"), errors="coerce", utc=True)
            if pd.isna(tipoff) or not game.get("gameId"):
                continue
            rows.append(
                {
                    "game_id": game["gameId"],
                    "season_year": season_year,
                    "game_date": pd.to_datetime(game_date).date(),
                    "tipoff_utc": tipoff,
                    "tipoff_et": game.get("gameEt"),
                    "team_home": (game.get("homeTeam") or {}).get("teamTricode"),
                    "team_away": (game.get("awayTeam") or {}).get("teamTricode"),
                    "arena_name": None,
                    "arena_city": None,
                    "game_status": game.get("gameStatusText"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    return pd.DataFrame(rows)[SCHEDULE_COLUMNS]


def fetch_schedules(season_years: list[int], *, timeout: int = 30) -> pd.DataFrame:
    """Concatenate :func:`fetch_season_schedule` across seasons."""
    frames = [
        fetch_season_schedule(season_year, timeout=timeout)
        for season_year in season_years
    ]
    if not frames:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
