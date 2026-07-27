import pandas as pd
import pytest
from nba_ou.data_processing.travel.travel_processing import (
    add_rolling_distances,
    compute_travel_features,
)


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


def test_rolling_travel_includes_current_trip_and_left_boundary():
    team_log = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3"],
            "GAME_DATE": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "TEAM_ID": ["team-1"] * 3,
            "TRAVEL_KM": [0.0, 100.0, 200.0],
        }
    )

    result = add_rolling_distances(team_log).set_index("GAME_ID")

    assert result.loc["g2", "KM_LAST_1_DAYS"] == pytest.approx(100.0)
    assert result.loc["g3", "KM_LAST_1_DAYS"] == pytest.approx(300.0)
