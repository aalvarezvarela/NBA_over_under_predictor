import argparse
from pathlib import Path

from nba_ou.postgre_db.sportsbook_matchup_history.create_db.create_sportsbook_matchup_history_db import (
    build_and_load_sportsbook_matchup_history_from_csvs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHUP_HISTORY_ROOT = PROJECT_ROOT / "data" / "sbr_line_history"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create/update sportsbook matchup-history table from local SBR CSV files."
        )
    )
    parser.add_argument(
        "--matchup-history-root",
        type=Path,
        default=DEFAULT_MATCHUP_HISTORY_ROOT,
        help="Root containing season folders with matchup_records/*_matchup_records.csv files.",
    )
    parser.add_argument(
        "--season-dir-glob",
        default="*",
        help="Season folder glob under --matchup-history-root, e.g. '2025-26'.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate the sportsbook matchup-history table before loading CSV data.",
    )
    parser.add_argument(
        "--strict-game-id-match",
        action="store_true",
        help="Fail if any CSV row cannot be matched to an NBA game_id.",
    )

    args = parser.parse_args()

    results = build_and_load_sportsbook_matchup_history_from_csvs(
        args.matchup_history_root,
        season_dir_glob=args.season_dir_glob,
        drop_existing=args.drop_existing,
        strict_game_id_match=args.strict_game_id_match,
    )

    print("Sportsbook matchup-history load summary:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
