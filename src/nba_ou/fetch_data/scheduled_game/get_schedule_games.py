from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2

EASTERN_TZ = ZoneInfo("America/New_York")


def nba_api_schedule_games(date):
    """
    Fetches NBA scheduled games for a given date.

    Parameters:
    date (str): The date in 'YYYY-MM-DD' format.

    Returns:
    DataFrame: A DataFrame containing scheduled games for the given date.
    """
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")

    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

    # Fetch games
    scoreboard_v2 = ScoreboardV2(game_date=date)

    games = scoreboard_v2.get_data_frames()[0]

    return games


def filter_started_games(
    games: pd.DataFrame, now_et: pd.Timestamp | datetime | None = None
) -> pd.DataFrame:
    """
    Keep only games that have not started yet based on GAME_TIME in Eastern Time.
    """
    if games.empty or "GAME_TIME" not in games.columns:
        return games

    current_time_et = pd.Timestamp.now(tz=EASTERN_TZ)
    if now_et is not None:
        current_time_et = pd.Timestamp(now_et)
        if current_time_et.tzinfo is None:
            current_time_et = current_time_et.tz_localize(EASTERN_TZ)
        else:
            current_time_et = current_time_et.tz_convert(EASTERN_TZ)

    upcoming_games = games[games["GAME_TIME"] > current_time_et].copy()

    filtered_count = len(games) - len(upcoming_games)
    if filtered_count > 0:
        print(
            f"Excluded {filtered_count} scheduled game(s) that already started "
            f"as of {current_time_et}."
        )

    return upcoming_games


def get_schedule_games(date_to_predict: str) -> pd.DataFrame:
    """
    Fetch scheduled NBA games for a specific date.
    Args:
        date_to_predict (str): Date in 'YYYY-MM-DD' format
    Returns:
        pd.DataFrame: DataFrame containing scheduled games
    """
    games = nba_api_schedule_games(date_to_predict)
    if games.empty:
        print("No games found for the specified date.")
        raise ValueError(
            "No games found for the specified date."
        )  # Return empty DataFrame if no games found

    # Extract just the date portion (first 10 chars: YYYY-MM-DD) and combine with time
    games["GAME_TIME"] = pd.to_datetime(
        games["GAME_DATE_EST"].astype(str).str[:10] + " " + games["GAME_STATUS_TEXT"],
        format="%Y-%m-%d %I:%M %p ET",
        errors="coerce",
    )

    # Filter out games with invalid GAME_TIME
    invalid_game_time = games[games["GAME_TIME"].isna()]
    if not invalid_game_time.empty:
        print("\n" + "==" * 30)
        print("WARNING: Dropping games with invalid GAME_TIME:")
        print("==" * 30)
        print(
            invalid_game_time[
                [
                    "GAME_ID",
                    "GAME_DATE_EST",
                    "GAME_STATUS_TEXT",
                    "HOME_TEAM_ID",
                    "VISITOR_TEAM_ID",
                ]
            ]
        )
        print("==" * 30 + "\n")
        games = games[games["GAME_TIME"].notna()].copy()

    # Make it timezone-aware (Eastern Time)
    games["GAME_TIME"] = games["GAME_TIME"].dt.tz_localize(
        EASTERN_TZ, ambiguous="infer", nonexistent="shift_forward"
    )
    games = filter_started_games(games)
    return games
