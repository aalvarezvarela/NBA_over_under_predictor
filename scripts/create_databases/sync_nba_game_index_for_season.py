import argparse

from nba_ou.postgre_db.game_time_index.sync_game_time_index import (
    sync_game_time_index_for_season,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create or update the Supabase NBA game index for a single season "
            "using game IDs already stored in nba_games."
        )
    )
    parser.add_argument(
        "season",
        help="Season to sync, for example 2024 or 2024-25.",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Refetch game payloads even when the season already has indexed values.",
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--strict-fetch",
        action="store_true",
        help="Abort immediately when one game payload fetch fails.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch the first N missing games. Useful for validation.",
    )

    args = parser.parse_args()
    summary = sync_game_time_index_for_season(
        args.season,
        refresh_all=args.refresh_all,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        strict_fetch=args.strict_fetch,
        limit=args.limit,
    )

    print("NBA game index sync summary:")
    for key, value in summary.as_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
