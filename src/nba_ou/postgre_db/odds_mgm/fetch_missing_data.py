import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from nba_ou.config.settings import Settings
from nba_ou.fetch_data.fetch_odds_data.get_odds_date import get_events_for_date
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_mgm,
)
from nba_ou.postgre_db.odds_mgm.create_db.create_odds_mgm_db import (
    load_games_for_season_years,
)
from psycopg import sql
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "TheRundownApi_data"


def _as_season_years(season_years: Iterable[int]) -> list[int]:
    return sorted({int(y) for y in season_years})


def load_odds_mgm_game_ids(season_years: Iterable[int]) -> set[str]:
    schema = get_schema_name_odds_mgm()
    table = schema
    season_years = _as_season_years(season_years)

    if not season_years:
        return set()

    conn = connect_nba_db()
    try:
        query_obj = sql.SQL(
            """
            SELECT DISTINCT game_id
            FROM {}.{}
            WHERE season_year = ANY(%s)
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))
        query = query_obj.as_string(conn)
        df = pd.read_sql_query(query, conn, params=(season_years,))
    finally:
        conn.close()

    return set(df["game_id"].dropna().astype(str))


def get_missing_games_df(season_years: Iterable[int]) -> pd.DataFrame:
    games_df = load_games_for_season_years(season_years)
    if games_df.empty:
        return pd.DataFrame()

    odds_game_ids = load_odds_mgm_game_ids(season_years)
    if not odds_game_ids:
        return games_df.drop_duplicates(subset=["game_id", "game_date"]).copy()

    games_df = games_df.copy()
    games_df["game_id"] = games_df["game_id"].astype(str)
    missing_mask = ~games_df["game_id"].isin(odds_game_ids)
    return games_df.loc[missing_mask].drop_duplicates(subset=["game_id", "game_date"])


def get_missing_game_dates(missing_games_df: pd.DataFrame) -> list[str]:
    if missing_games_df.empty:
        return []

    game_dates = pd.to_datetime(
        missing_games_df["game_date"], errors="coerce"
    ).dt.normalize()

    date_set: set[str] = set()
    for dt in game_dates.dropna():
        date_set.add(dt.date().isoformat())
        date_set.add((dt + pd.Timedelta(days=1)).date().isoformat())

    cutoff = pd.Timestamp("2023-03-15")
    filtered = [d for d in date_set if pd.Timestamp(d) >= cutoff]
    missing_dates = sorted(filtered)
    return missing_dates


def iter_missing_json_dates(
    missing_dates: Iterable[str], json_dir: str | Path
) -> list[str]:
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    missing_json_dates: list[str] = []
    for date_str in missing_dates:
        json_path = json_dir / f"{date_str}.json"
        if not json_path.exists():
            missing_json_dates.append(date_str)
    return missing_json_dates


def fetch_events_for_date(
    date_str: str,
    base_url: str,
    headers: dict[str, str],
    *,
    sleep_seconds: float = 0.25,
) -> list[dict]:
    events = get_events_for_date(
        sport_id=4,
        date=date_str,
        base_url=base_url,
        headers=headers,
    )
    time.sleep(sleep_seconds)
    return events


def save_events_json(events: list[dict], json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    return json_path


def fetch_and_save_missing_json(
    missing_json_dates: Iterable[str],
    json_dir: str | Path,
    base_url: str,
    headers: dict[str, str],
    *,
    skip_if_exists: bool = True,
) -> dict[str, int]:
    json_dir = Path(json_dir)
    results = {"fetched": 0, "skipped_empty": 0, "errors": 0}

    for date_str in tqdm(missing_json_dates):
        json_path = json_dir / f"{date_str}.json"
        if skip_if_exists and json_path.exists():
            print(f"Skipping existing JSON for {date_str}")
            continue
        try:
            events = fetch_events_for_date(
                date_str=date_str, base_url=base_url, headers=headers
            )
            if not events:
                results["skipped_empty"] += 1
                continue

            save_events_json(events, json_path)
            time.sleep(0.25)
            results["fetched"] += 1
        except Exception as exc:
            results["errors"] += 1
            print(f"Error fetching {date_str}: {exc}")

    return results


def fetch_missing_odds_mgm_json(
    season_years: Iterable[int],
    json_dir: str | Path = DEFAULT_JSON_DIR,
) -> dict[str, object]:
    settings = Settings()
    base_url = settings.odds_base_url
    headers = {
        "x-rapidapi-key": settings.odds_api_key,
        "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
    }

    missing_games_df = get_missing_games_df(season_years)
    missing_dates = get_missing_game_dates(missing_games_df)
    missing_json_dates = iter_missing_json_dates(missing_dates, json_dir)

    results = fetch_and_save_missing_json(
        missing_json_dates=missing_json_dates,
        json_dir=json_dir,
        base_url=base_url,
        headers=headers,
    )

    return {
        "season_years": _as_season_years(season_years),
        "missing_games": len(missing_games_df),
        "missing_dates": len(missing_dates),
        "missing_json_dates": len(missing_json_dates),
        "results": results,
        "json_dir": str(Path(json_dir)),
    }


if __name__ == "__main__":
    season_years = [2022, 2023, 2024, 2025]
    summary = fetch_missing_odds_mgm_json(season_years)

    print("\nMissing odds MGM JSON fetch summary")
    print(f"Season years: {summary['season_years']}")
    print(f"Missing games: {summary['missing_games']}")
    print(f"Missing game dates: {summary['missing_dates']}")
    print(f"Missing json dates: {summary['missing_json_dates']}")
    print(f"Fetched json: {summary['results']['fetched']}")
    print(f"Skipped (no events): {summary['results']['skipped_empty']}")
    print(f"Errors: {summary['results']['errors']}")
    print(f"JSON dir: {summary['json_dir']}")
