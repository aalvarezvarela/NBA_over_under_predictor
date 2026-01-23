from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.data_preparation.referees.process_refs_scheduled_game import (
    process_scheduled_referee_assignments,
)
from nba_ou.data_preparation.scheduled_games.manage_injury_data import (
    process_injury_data,
)
from nba_ou.fetch_data.fetch_odds_data.get_odds_date import process_odds_date
from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    retrieve_injury_report_as_df,
)
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games


def get_all_info_for_scheduled_games(
    date_to_predict: str,
    nba_injury_reports_url,
    save_reports_path,
    odds_api_key,
    odds_base_url,
) -> pd.DataFrame:
    if not date_to_predict:
        date_to_predict = pd.Timestamp.now(tz=ZoneInfo("US/Eastern")).strftime(
            "%Y-%m-%d"
        )
    # First Get the games itself
    scheduled_games = get_schedule_games(date_to_predict)
    if scheduled_games.empty:
        print(f"No scheduled games found for {date_to_predict}")
        raise ValueError("No scheduled games found")
    # Then get refs
    df_referees_scheduled = process_scheduled_referee_assignments(scheduled_games)
    # Then Injuries
    injury_report_df = retrieve_injury_report_as_df(
        nba_injury_reports_url, reports_path=save_reports_path
    )

    injury_dict_scheduled, games_not_updated = process_injury_data(
        scheduled_games, injury_report_df
    )

    if len(games_not_updated) == len(scheduled_games):
        raise ValueError("No games were updated with injury data")

    headers = {
        "x-rapidapi-key": odds_api_key,
        "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
    }

    # Then New odds
    df_today = process_odds_date(
        date_to_predict,
        BASE_URL=odds_base_url,
        HEADERS=headers,
        is_today=True,
    )
    df_tomorrow = process_odds_date(
        (pd.to_datetime(date_to_predict) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        BASE_URL=odds_base_url,
        HEADERS=headers,
        is_today=True,
    )
    df_odds_scheduled = pd.concat([df_today, df_tomorrow], ignore_index=True)
    # Fitler and avoid problems
    df_odds_scheduled.sort_values(by="game_date", inplace=True, ascending=False)
    df_odds_scheduled.reset_index(drop=True, inplace=True)
    df_odds_scheduled = df_odds_scheduled[
        df_odds_scheduled["most_common_total_line"] > 150
    ]
    df_odds_scheduled = df_odds_scheduled.drop_duplicates(
        subset=("game_date", "team_home", "team_away"), keep="first"
    )

    return (
        scheduled_games,
        df_referees_scheduled,
        injury_dict_scheduled,
        df_odds_scheduled,
    )
