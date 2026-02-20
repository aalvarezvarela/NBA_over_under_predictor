"""
NBA Over/Under Predictor - Historical Database Backfill Script

Backfills NBA games and player game logs into PostgreSQL by iterating season-by-season.
Uses the existing `update_database` function (which fetches only missing game_ids).

Covers seasons from 2005-06 through 2024-25, i.e. games played in calendar years 2006..2025.
"""

from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases

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
        "--only-new-games",
        action="store_true",
        default=False,
        help="Only update odds and refs/injuries if new games are found",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=False,
        help="Run Playwright in headed mode (default uses config.ini)",
    )

    args = parser.parse_args()
    headless = False if args.headed else None

    mode_text = (
        "config default" if headless is None else "headless" if headless else "headed"
    )
    print(f"Playwright mode: {mode_text}")

    update_all_databases(
        start_season_year=args.start,
        end_season_year=args.end,
        only_new_games=args.only_new_games,
        headless=headless,
    )
