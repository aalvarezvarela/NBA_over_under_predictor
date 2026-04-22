import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games

DEFAULT_TIMEZONE = "US/Pacific"
DEFAULT_OUTPUT_KEY = "has_scheduled_games"


def resolve_date_to_predict(date_to_predict: str | None, timezone_name: str) -> str:
    if date_to_predict:
        return date_to_predict
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%d")


def write_github_output(output_key: str, has_scheduled_games: bool) -> None:
    github_output_path = os.getenv("GITHUB_OUTPUT")
    if not github_output_path:
        return

    with open(github_output_path, "a", encoding="utf-8") as github_output:
        github_output.write(
            f"{output_key}={'true' if has_scheduled_games else 'false'}\n"
        )


def check_scheduled_games(
    date_to_predict: str | None = None,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    output_key: str = DEFAULT_OUTPUT_KEY,
) -> bool:
    resolved_date = resolve_date_to_predict(date_to_predict, timezone_name)
    scheduled_games = get_schedule_games(resolved_date)
    has_scheduled_games = not scheduled_games.empty

    print(f"Checked scheduled games for {resolved_date}")
    if has_scheduled_games:
        print(
            f"Found {len(scheduled_games)} scheduled game(s). "
            "Continuing with downstream steps."
        )
    else:
        print(
            f"No scheduled games found for {resolved_date}. "
            "Skipping downstream steps."
        )

    write_github_output(output_key, has_scheduled_games)
    return has_scheduled_games


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether there are scheduled NBA games for a date."
    )
    parser.add_argument(
        "--date",
        dest="date_to_predict",
        help="Date to check in YYYY-MM-DD format. Defaults to today's date.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone used when --date is omitted. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument(
        "--output-key",
        default=DEFAULT_OUTPUT_KEY,
        help=(
            "GitHub Actions output key written to GITHUB_OUTPUT when available. "
            f"Default: {DEFAULT_OUTPUT_KEY}."
        ),
    )
    args = parser.parse_args()

    check_scheduled_games(
        date_to_predict=args.date_to_predict,
        timezone_name=args.timezone,
        output_key=args.output_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
