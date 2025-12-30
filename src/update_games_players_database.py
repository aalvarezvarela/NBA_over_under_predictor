"""
NBA Over/Under Predictor - Historical Database Backfill Script

Backfills NBA games and player game logs into PostgreSQL by iterating season-by-season.
Uses the existing `update_database` function (which fetches only missing game_ids).

Covers seasons from 2005-06 through 2024-25, i.e. games played in calendar years 2006..2025.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

# Import your existing updater
# Adjust this import to your project structure if needed.
from fetch_data.manage_games_database.update_database import update_database


def season_start_date(season_start_year: int) -> datetime:
    """
    Return a date that falls inside the NBA season that starts in `season_start_year`.
    Using Dec 1 avoids edge cases around Oct/Nov logic in get_nba_season_nullable.
    Example: season_start_year=2005 -> date 2005-12-01 -> season "2005-06"
    """
    return datetime(season_start_year, 12, 1)


def backfill_seasons(
    data_folder: str | Path,
    start_season_year: int = 2005,
    end_season_year: int = 2025,
    save_csv: bool = True,
    sleep_seconds_between_seasons: float = 2.0,
) -> None:
    """
    Backfill from `start_season_year`-YY through `end_season_year`-(YY+1).

    Args:
        data_folder: Folder where seasonal CSVs are stored (optional feature of update_database).
        start_season_year: First season start year (2005 -> 2005-06).
        end_season_year: Last season start year (2024 -> 2024-25).
        save_csv: Whether to also persist CSV snapshots per season.
        sleep_seconds_between_seasons: Small pause to be polite with the NBA API.
    """
    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    for y in range(start_season_year, end_season_year + 1):
        date_in_season = season_start_date(y)
        print(
            f"\n=== Backfilling season starting {y} (date={date_in_season.date()}) ==="
        )

        limit_reached = update_database(
            database_folder=str(data_folder),
            date=date_in_season,
            save_csv=save_csv,
        )

        if limit_reached:
            print(
                "\n⚠️ API limit reached or throttling detected. "
                "Stopping backfill now. Re-run this script later to continue."
            )
            return

        time.sleep(sleep_seconds_between_seasons)

    print("\n✅ Backfill completed for all requested seasons.")


if __name__ == "__main__":
    DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data/"
    )
    backfill_seasons(
        data_folder=DATA_FOLDER,
        start_season_year=2006,  # 2005-06 -> games in 2006
        end_season_year=2024,  # 2024-25 -> games in 2025
        save_csv=True,
        sleep_seconds_between_seasons=2.0,
    )
