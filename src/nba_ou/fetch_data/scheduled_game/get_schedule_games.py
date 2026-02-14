from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2


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
        "US/Eastern", ambiguous="infer", nonexistent="shift_forward"
    )
    return games
