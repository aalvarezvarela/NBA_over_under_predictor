"""
NBA Over/Under Predictor - Historical Database Backfill Script

Backfills NBA games and player game logs into PostgreSQL by iterating season-by-season.
Uses the existing `update_database` function (which fetches only missing game_ids).

Covers seasons from 2005-06 through 2024-25, i.e. games played in calendar years 2006..2025.
"""

import time

from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.live_games.live_games import get_live_game_ids
from nba_ou.postgre_db.games.update_games.update_database import (
    update_team_players_database,
)
from nba_ou.postgre_db.odds.update_odds.update_odds_database import update_odds_database

# def season_start_date(season_start_year: int) -> datetime:
#     """
#     Return a date that falls inside the NBA season that starts in `season_start_year`.
#     Using Oct 1 avoids edge cases around Oct/Nov logic in get_nba_season_nullable.
#     Example: season_start_year=2005 -> date 2005-10-01 -> season "2005-06"
#     """
# from datetime import datetime

# return datetime(season_start_year, 10, 1)


def update_all_databases(
    start_season_year: int = 2006,
    end_season_year: int = 2025,
    sleep_seconds_between_seasons: float = 2.0,
    games_id_to_exclude: list = None,
) -> None:
    """
    Backfill from `start_season_year`-YY through `end_season_year`-(YY+1).

    Args:
        data_folder: Folder where seasonal CSVs are stored (optional feature of update_database).
        start_season_year: First season start year (2005 -> 2005-06).
        end_season_year: Last season start year (2024 -> 2024-25).
        sleep_seconds_between_seasons: Small pause to be polite with the NBA API.
    """
    for y in range(start_season_year, end_season_year + 1):
        season_year = str(y)
        season_nullable = f"{season_year}-{str(y + 1)[-2:]}"
        print(f"\n=== Backfilling season starting {season_year} season) ===")

        limit_reached = update_team_players_database(
            season_year=season_year,
            games_id_to_exclude=games_id_to_exclude,
        )

        if limit_reached:
            print(
                "\n⚠️ API limit reached or throttling detected. "
                "Stopping backfill now. Re-run this script later to continue."
            )
            raise RuntimeError("API limit reached during backfill.")

        print(f"\n--- Updating odds for season {y} ---")

        update_odds_database(
            season_year=season_year,
            ODDS_API_KEY=SETTINGS.odds_api_key,
            BASE_URL=SETTINGS.odds_base_url,
            check_missing_by_game=True,
            save_pickle=SETTINGS.odds_save_pickle,
            pickle_path=SETTINGS.odds_pickle_path,
        )
        print(f"✓ Odds data updated for season {y}")

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
    update_all_databases(
        start_season_year=args.start,
        end_season_year=args.end,
        games_id_to_exclude=ids_to_exclude,
    )
