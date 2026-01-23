"""
NBA Over/Under Predictor - Historical Database Backfill Script

Backfills NBA games and player game logs into PostgreSQL by iterating season-by-season.
Uses the existing `update_database` function (which fetches only missing game_ids).

Covers seasons from 2005-06 through 2024-25, i.e. games played in calendar years 2006..2025.
"""

import time
from datetime import datetime

from nba_ou.config.settings import SETTINGS as settings
from nba_ou.fetch_data.live_games.live_games import get_live_game_ids
from nba_ou.postgre_db.games.update_games.update_database import (
    update_team_players_database,
)
from nba_ou.postgre_db.odds.update_odds.update_odds_database import update_odds_database


def season_start_date(season_start_year: int) -> datetime:
    """
    Return a date that falls inside the NBA season that starts in `season_start_year`.
    Using Dec 1 avoids edge cases around Oct/Nov logic in get_nba_season_nullable.
    Example: season_start_year=2005 -> date 2005-12-01 -> season "2005-06"
    """
    return datetime(season_start_year, 12, 1)


def backfill_seasons(
    start_season_year: int = 2006,
    end_season_year: int = 2025,
    sleep_seconds_between_seasons: float = 2.0,
    update_odds_data: bool = False,
    games_id_to_exclude: list = None,
) -> None:
    """
    Backfill from `start_season_year`-YY through `end_season_year`-(YY+1).

    Args:
        data_folder: Folder where seasonal CSVs are stored (optional feature of update_database).
        start_season_year: First season start year (2005 -> 2005-06).
        end_season_year: Last season start year (2024 -> 2024-25).
        sleep_seconds_between_seasons: Small pause to be polite with the NBA API.
        update_odds_data: Whether to also backfill odds data for each season.
    """
    for y in range(start_season_year, end_season_year + 1):
        date_in_season = season_start_date(y)
        print(
            f"\n=== Backfilling season starting {y} (date={date_in_season.date()}) ==="
        )

        limit_reached = update_team_players_database(
            games_id_to_exclude=games_id_to_exclude,
            date=date_in_season,
        )

        if limit_reached:
            print(
                "\n⚠️ API limit reached or throttling detected. "
                "Stopping backfill now. Re-run this script later to continue."
            )
            raise RuntimeError("API limit reached during backfill.")

        # Update odds data if requested
        if update_odds_data:
            print(f"\n--- Updating odds for season {y} ---")
            try:
                season_nullable = f"{y}-{str(y + 1)[-2:]}"
                update_odds_database(
                    season_to_download=season_nullable,
                    ODDS_API_KEY=settings.odds_api_key,
                    BASE_URL=settings.odds_base_url,
                    save_pickle=settings.odds_save_pickle,
                    pickle_path=settings.odds_pickle_path,
                )
                print(f"✓ Odds data updated for season {y}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to update odds for season {y}: {e}")
                print("Continuing with next season...")

        time.sleep(sleep_seconds_between_seasons)

    print("\n✅ Backfill completed for all requested seasons.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill NBA games and player logs by season."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2025,
        help="First season start year (e.g. 2023 for 2023-24)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2025,
        help="Last season start year (e.g. 2025 for 2025-26)",
    )
    parser.add_argument(
        "--update-odds",
        action="store_true",
        help="If set, also backfill odds data for each season (default: False)",
    )

    args = parser.parse_args()

    ids_to_exclude = get_live_game_ids()
    backfill_seasons(
        start_season_year=args.start,
        end_season_year=args.end,
        update_odds_data=args.update_odds,
        games_id_to_exclude=ids_to_exclude,
    )
