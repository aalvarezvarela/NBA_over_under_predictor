import pandas as pd

from nba_ou.data_processing.team.records import add_team_record_before_game


def test_add_team_record_before_game_does_not_shift_across_groups():
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-03",
                ]
            ),
            "WL": ["W", "W", "L", "W"],
            "SEASON_TYPE": ["Regular Season"] * 4,
            "SEASON_ID": ["22025", "22025", "22025", "22025"],
            "TEAM_ID": ["A", "A", "B", "B"],
        }
    )

    result = add_team_record_before_game(df)

    team_a = result[result["TEAM_ID"] == "A"].sort_values("GAME_DATE")
    team_b = result[result["TEAM_ID"] == "B"].sort_values("GAME_DATE")

    assert team_a["WINS_BEFORE_THIS_GAME"].tolist() == [0, 1]
    assert team_a["TEAM_RECORD_BEFORE_GAME"].tolist() == [0.0, 1.0]
    assert team_b["WINS_BEFORE_THIS_GAME"].tolist() == [0, 0]
    assert team_b["TEAM_RECORD_BEFORE_GAME"].tolist() == [0.0, 0.0]
