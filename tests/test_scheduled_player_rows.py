import pandas as pd

from nba_ou.data_preparation.players.players_statistics import (
    precompute_cumulative_avg_stat,
)
from nba_ou.data_preparation.scheduled_games.merge_scheduled_with_existing_data import (
    standardize_and_merge_scheduled_games_to_players_data,
)


def test_scheduled_player_rows_keep_historical_season_bucket() -> None:
    df_players = pd.DataFrame(
        [
            {
                "GAME_ID": "0022500900",
                "GAME_DATE": "2026-03-15",
                "SEASON_ID": "22025",
                "SEASON_YEAR": 2025,
                "TEAM_ID": "1610612747",
                "PLAYER_ID": "201939",
                "PLAYER_NAME": "S. Curry",
                "START_POSITION": "G",
                "MIN": 35.0,
                "PTS": 25,
            }
        ]
    )

    scheduled_games = pd.DataFrame(
        [
            {
                "GAME_ID": "0022500999",
                "SEASON": "2025",
                "GAME_DATE_EST": "2026-03-17T00:00:00",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            }
        ]
    )

    scheduled_rows = standardize_and_merge_scheduled_games_to_players_data(
        scheduled_games, df_players
    )

    scheduled_row = scheduled_rows.loc[scheduled_rows["PLAYER_ID"] == "201939"].iloc[0]
    assert scheduled_row["SEASON_YEAR"] == 2025
    assert scheduled_rows["SEASON_YEAR"].dtype == df_players["SEASON_YEAR"].dtype
    assert scheduled_row["SEASON_ID"] == "22025"

    combined = pd.concat([df_players, scheduled_rows], ignore_index=True, sort=False)
    with_cum_avg = precompute_cumulative_avg_stat(combined, stat_col="PTS")

    scheduled_with_avg = with_cum_avg.loc[
        (with_cum_avg["GAME_ID"] == "0022500999")
        & (with_cum_avg["PLAYER_ID"] == "201939")
    ].iloc[0]
    assert scheduled_with_avg["PTS_CUM_AVG"] == 25
