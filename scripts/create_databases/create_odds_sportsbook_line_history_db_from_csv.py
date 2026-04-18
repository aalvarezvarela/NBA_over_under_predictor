import argparse
from pathlib import Path

from nba_ou.postgre_db.odds_sportsbook_line_history.create_db.create_odds_sportsbook_line_history_db import (
    build_and_load_odds_sportsbook_line_history_from_csvs,
)
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (
    LINE_HISTORY_MARKETS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LINE_HISTORY_ROOT = PROJECT_ROOT / "data" / "sbr_line_history"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create/update sportsbook line-history tables from local SBR CSV files."
        )
    )
    parser.add_argument(
        "--line-history-root",
        type=Path,
        default=DEFAULT_LINE_HISTORY_ROOT,
        help="Root containing season folders with line_history/*_line_history.csv files.",
    )
    parser.add_argument(
        "--season-dir-glob",
        default="*",
        help="Season folder glob under --line-history-root, e.g. '2025-26'.",
    )
    parser.add_argument(
        "--market",
        action="append",
        choices=LINE_HISTORY_MARKETS + ["total", "moneyline", "spread"],
        default=None,
        help=(
            "Market to load. Can be provided multiple times. "
            "Defaults to totals, money_line, and point_spread."
        ),
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate target line-history tables before loading CSV data.",
    )
    parser.add_argument(
        "--strict-game-id-match",
        action="store_true",
        help="Fail if any CSV row cannot be matched to an NBA game_id.",
    )

    args = parser.parse_args()

    results = build_and_load_odds_sportsbook_line_history_from_csvs(
        args.line_history_root,
        season_dir_glob=args.season_dir_glob,
        markets=args.market,
        drop_existing=args.drop_existing,
        strict_game_id_match=args.strict_game_id_match,
    )

    print("Sportsbook line-history load summary:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
