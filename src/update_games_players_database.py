"""
NBA Over/Under Predictor - Historical Database Backfill Script

Backfills NBA games and player game logs into PostgreSQL by iterating season-by-season.
Uses the existing `update_database` function (which fetches only missing game_ids).

Covers seasons from 2005-06 through 2024-25, i.e. games played in calendar years 2006..2025.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # if script is inside src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import settings
from fetch_data.manage_games_database.update_database import update_database
from fetch_data.manage_odds_data.update_odds import update_odds


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
    data_folder: str | Path = None,
    sleep_seconds_between_seasons: float = 2.0,
    update_odds_data: bool = False,
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
    # Setup odds folder if needed
    odds_folder = None
    
    for y in range(start_season_year, end_season_year + 1):
        date_in_season = season_start_date(y)
        print(
            f"\n=== Backfilling season starting {y} (date={date_in_season.date()}) ==="
        )

        limit_reached = update_database(
            database_folder=str(data_folder) if data_folder else None,
            date=date_in_season,
            save_csv=False,
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
                update_odds(
                    date_to_predict=date_in_season.strftime("%Y-%m-%d"),
                    odds_folder=str(odds_folder) if odds_folder else None,
                    ODDS_API_KEY=settings.odds_api_key,
                    BASE_URL=settings.odds_base_url,
                    save_csv=False,
                )
                print(f"✓ Odds data updated for season {y}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to update odds for season {y}: {e}")
                print("Continuing with next season...")

        time.sleep(sleep_seconds_between_seasons)

    print("\n✅ Backfill completed for all requested seasons.")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Backfill NBA games and player logs by season."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2023,
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
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Folder to store seasonal CSVs",
    )

    args = parser.parse_args()

    backfill_seasons(
        start_season_year=args.start,
        end_season_year=args.end,
        data_folder=args.data_folder,
        update_odds_data=args.update_odds,
    )
