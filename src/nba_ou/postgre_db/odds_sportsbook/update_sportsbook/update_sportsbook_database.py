import argparse
import asyncio
from datetime import date

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    merge_sportsbook_with_games,
)
from nba_ou.postgre_db.odds_sportsbook.create_db.create_odds_sportsbook_db import (
    create_odds_sportsbook_table,
    upsert_odds_sportsbook_df,
)
from nba_ou.postgre_db.odds_sportsbook.update_sportsbook.scrape_sportsbook import (
    scrape_sportsbook_days,
)
from nba_ou.postgre_db.odds_sportsbook.update_sportsbook.update_database_utils import (
    get_dates_for_game_ids,
    get_missing_game_ids_to_scrape,
    load_games_for_sportsbook_update,
    select_target_game_ids,
)


def _normalize_game_id(df: pd.DataFrame, col: str = "game_id") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str)
    return out


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize team_home and team_away columns using TEAM_NAME_STANDARDIZATION."""
    out = df.copy()
    if "team_home" in out.columns:
        out["team_home"] = out["team_home"].map(
            lambda x: TEAM_NAME_STANDARDIZATION.get(x, x) if pd.notna(x) else x
        )
    if "team_away" in out.columns:
        out["team_away"] = out["team_away"].map(
            lambda x: TEAM_NAME_STANDARDIZATION.get(x, x) if pd.notna(x) else x
        )
    return out


def update_odds_sportsbook_database(
    *,
    last_n_games: int | None = None,
    season_year: str | int | None = None,
    headless: bool = True,
) -> dict[str, int]:
    """
    Incrementally update odds_sportsbook DB by scraping only missing GAME_IDs.

    Flow:
    1) Load all games from DB (optionally filter by season)
    2) Select target GAME_IDs (optionally latest N)
    3) Find missing GAME_IDs in odds_sportsbook table
    4) Scrape only corresponding dates from sportsbookreview
    5) Merge scraped rows with games DB to resolve canonical GAME_ID
    6) Upsert into odds_sportsbook table
    """
    if not create_odds_sportsbook_table(drop_existing=False):
        raise RuntimeError("Failed to create/validate odds_sportsbook table.")

    games_df = load_games_for_sportsbook_update(season_year=season_year)
    if games_df.empty:
        print("No games found in games database for sportsbook update.")
        return {
            "target_game_ids": 0,
            "missing_game_ids": 0,
            "scraped_rows": 0,
            "merged_rows": 0,
            "inserted_rows": 0,
        }

    target_game_ids = select_target_game_ids(games_df, last_n_games=last_n_games)
    missing_game_ids = get_missing_game_ids_to_scrape(target_game_ids)

    print(f"Target games: {len(target_game_ids)}")
    print(f"Missing sportsbook games: {len(missing_game_ids)}")

    if not missing_game_ids:
        print("No missing game IDs to scrape. Database is up to date for target games.")
        return {
            "target_game_ids": len(target_game_ids),
            "missing_game_ids": 0,
            "scraped_rows": 0,
            "merged_rows": 0,
            "inserted_rows": 0,
        }

    scrape_dates_ts = get_dates_for_game_ids(games_df, missing_game_ids)
    scrape_dates: list[date] = [d.date() for d in scrape_dates_ts]

    print(f"Scraping sportsbook for {len(scrape_dates)} date(s)...")
    scraped_df = asyncio.run(scrape_sportsbook_days(scrape_dates, headless=headless))

    if scraped_df.empty:
        print("No sportsbook rows scraped for missing games.")
        return {
            "target_game_ids": len(target_game_ids),
            "missing_game_ids": len(missing_game_ids),
            "scraped_rows": 0,
            "merged_rows": 0,
            "inserted_rows": 0,
        }

    scraped_df = _normalize_game_id(scraped_df)
    scraped_df = _normalize_team_names(scraped_df)
    games_df = _normalize_game_id(games_df)

    merged_df = merge_sportsbook_with_games(scraped_df, games_df)
    merged_df = (
        merged_df.dropna(subset=["game_id"])
        if "game_id" in merged_df
        else pd.DataFrame()
    )

    if merged_df.empty:
        print("Scraped rows could not be mapped to game_id.")
        return {
            "target_game_ids": len(target_game_ids),
            "missing_game_ids": len(missing_game_ids),
            "scraped_rows": len(scraped_df),
            "merged_rows": 0,
            "inserted_rows": 0,
        }

    merged_df = _normalize_game_id(merged_df)
    missing_set = set(missing_game_ids)
    merged_df = merged_df[merged_df["game_id"].isin(missing_set)]

    inserted_rows = upsert_odds_sportsbook_df(merged_df)

    print(f"Scraped rows: {len(scraped_df)}")
    print(f"Mapped rows: {len(merged_df)}")
    print(f"Inserted rows: {inserted_rows}")

    return {
        "target_game_ids": len(target_game_ids),
        "missing_game_ids": len(missing_game_ids),
        "scraped_rows": len(scraped_df),
        "merged_rows": len(merged_df),
        "inserted_rows": inserted_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Update odds_sportsbook DB by scraping missing game IDs from sportsbookreview"
        )
    )
    parser.add_argument(
        "--last-n-games",
        type=int,
        default=None,
        help="If provided, only check and update the latest N games from the games DB",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=2025,
        help="Optional season start year filter (e.g., 2025 for 2025-26 season)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=True,
        help="Run browser in headed mode for debugging",
    )

    args = parser.parse_args()

    results = update_odds_sportsbook_database(
        last_n_games=args.last_n_games,
        season_year=args.season_year,
        headless=not args.headed,
    )
    print("Update summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
