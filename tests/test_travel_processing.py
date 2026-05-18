import pandas as pd
from nba_ou.data_processing.travel.travel_processing import compute_travel_features


def test_compute_travel_features_adds_recent_jetlag_hours():
    df = pd.DataFrame(
        [
            {
                "GAME_ID": "001",
                "GAME_DATE": "2026-01-01",
                "TEAM_ID_TEAM_HOME": "BOS",
                "TEAM_CITY_TEAM_HOME": "Boston",
                "TEAM_ID_TEAM_AWAY": "LAL",
                "TEAM_CITY_TEAM_AWAY": "Los Angeles",
            },
            {
                "GAME_ID": "002",
                "GAME_DATE": "2026-01-03",
                "TEAM_ID_TEAM_HOME": "LAL",
                "TEAM_CITY_TEAM_HOME": "Los Angeles",
                "TEAM_ID_TEAM_AWAY": "BOS",
                "TEAM_CITY_TEAM_AWAY": "Boston",
            },
            {
                "GAME_ID": "003",
                "GAME_DATE": "2026-01-10",
                "TEAM_ID_TEAM_HOME": "BOS",
                "TEAM_CITY_TEAM_HOME": "Boston",
                "TEAM_ID_TEAM_AWAY": "LAL",
                "TEAM_CITY_TEAM_AWAY": "Los Angeles",
            },
        ]
    )

    result = compute_travel_features(df, log_scale=True)

    assert result.loc[0, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 0
    assert result.loc[0, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 0
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 3
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 3
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 0
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 0
