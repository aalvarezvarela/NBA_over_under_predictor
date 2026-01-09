"""
NBA Over/Under Predictor - Historical Refs and Injuries Database Backfill Script

Backfills NBA referees and injuries data into PostgreSQL by iterating season-by-season.
Uses the existing `update_refs_injuries_database` function (which fetches only missing game_ids).

Covers seasons from 2006-07 through 2024-25, i.e. games played in calendar years 2006..2025.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # if script is inside src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fetch_data.manage_refs_injury_database.update_refs_injuries_database import (
    update_refs_injuries_database,
)


def season_start_date(season_start_year: int) -> datetime:
    """
    Return a date that falls inside the NBA season that starts in `season_start_year`.
    Using Dec 1 avoids edge cases around Oct/Nov logic in get_nba_season_nullable.
    Example: season_start_year=2006 -> date 2006-12-01 -> season "2006-07"
    """
    return datetime(season_start_year, 12, 1)


def backfill_refs_injuries_seasons(
    start_season_year: int = 2006,
    end_season_year: int = 2025,
    injury_folder: str | Path = None,
    ref_folder: str | Path = None,
    sleep_seconds_between_seasons: float = 2.0,
    save_csv: bool = False,
) -> None:
    """
    Backfill referees and injuries data from `start_season_year`-YY through `end_season_year`-(YY+1).

    Args:
        start_season_year: First season start year (2006 -> 2006-07).
        end_season_year: Last season start year (2024 -> 2024-25).
        injury_folder: Folder where injury CSVs are stored.
        ref_folder: Folder where referee CSVs are stored.
        sleep_seconds_between_seasons: Small pause to be polite with the NBA API.
        save_csv: Whether to save CSV backups (default: False).
    """
    for y in range(start_season_year, end_season_year + 1):
        date_in_season = season_start_date(y)
        print(
            f"\n=== Backfilling refs/injuries for season starting {y} (date={date_in_season.date()}) ==="
        )

        try:
            limit_reached = update_refs_injuries_database(
                injury_folder=str(injury_folder) if injury_folder else None,
                ref_folder=str(ref_folder) if ref_folder else None,
                date=date_in_season,
                save_csv=save_csv,
            )

            if limit_reached:
                print(
                    "\n⚠️ API limit reached or throttling detected. "
                    "Stopping backfill now. Re-run this script later to continue."
                )
                raise RuntimeError("API limit reached during backfill.")

        except Exception as e:
            print(f"⚠️ Error backfilling season {y}: {e}")
            import traceback

            traceback.print_exc()
            raise e
            print("Continuing with next season...")

        time.sleep(sleep_seconds_between_seasons)

    print("\n✅ Backfill completed for all requested seasons.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill NBA referees and injuries data by season."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2024,
        help="First season start year (e.g. 2006 for 2006-07)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2025,
        help="Last season start year (e.g. 2025 for 2025-26)",
    )
    parser.add_argument(
        "--injury-folder",
        type=str,
        default="./data/injury_data",
        help="Folder to store injury CSVs",
    )
    parser.add_argument(
        "--ref-folder",
        type=str,
        default="./data/ref_data",
        help="Folder to store referee CSVs",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="If set, save CSV backups (default: False)",
    )

    args = parser.parse_args()

    backfill_refs_injuries_seasons(
        start_season_year=args.start,
        end_season_year=args.end,
        injury_folder=args.injury_folder,
        ref_folder=args.ref_folder,
        save_csv=args.save_csv,
    )
