import nba_ou.data_processing.team.rolling as rolling_module
import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    select_training_columns,
)
from nba_ou.data_processing.past_injuries.injury_effects import (
    add_top3_availability_effect_features_for_columns,
)
from nba_ou.data_processing.statistics.statistics import compute_rolling_stats
from nba_ou.data_processing.team.rolling import (
    COLS_TO_AVERAGE,
    OVERTIME_HISTORY_FEATURE_COLUMNS,
    _get_rolling_price_columns,
    compute_all_rolling_statistics,
)


def _availability_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [1],
            "GAME_DATE": ["2026-01-01"],
            "SEASON_YEAR": [2025],
            "TEAM_ID_TEAM_HOME": [10],
            "TEAM_ID_TEAM_AWAY": [20],
            "TOTAL_POINTS": [220.0],
            "TOTAL_LINE_bet365": [215.5],
            "HOME_PLAYER": [101],
            "AWAY_PLAYER": [201],
        }
    )


def _add_availability_features(include_detailed: bool) -> pd.DataFrame:
    return add_top3_availability_effect_features_for_columns(
        _availability_input(),
        injured_dict={},
        total_line_book="bet365",
        home_player_cols=("HOME_PLAYER",),
        away_player_cols=("AWAY_PLAYER",),
        out_prefix="TEST_AVAILABILITY",
        include_detailed_sample_size_features=include_detailed,
    )


def test_availability_defaults_to_compact_aggregate_schema():
    result = _add_availability_features(include_detailed=False)
    feature_columns = [
        column for column in result.columns if column.startswith("TEST_AVAILABILITY_")
    ]

    assert len(feature_columns) == 10
    assert "TEST_AVAILABILITY_HOME_SUM_N_TOTAL_GAMES" in feature_columns
    assert "TEST_AVAILABILITY_AWAY_SUM_N_TOTAL_GAMES" in feature_columns
    assert "TEST_AVAILABILITY_HOME_SUM_N_INJ_GAMES" not in feature_columns
    assert "TEST_AVAILABILITY_HOME_HAS_PLAYER_EFFECT" not in feature_columns


def test_availability_detailed_diagnostics_can_be_restored():
    result = _add_availability_features(include_detailed=True)
    feature_columns = [
        column for column in result.columns if column.startswith("TEST_AVAILABILITY_")
    ]

    assert len(feature_columns) == 18
    assert "TEST_AVAILABILITY_HOME_SUM_N_INJ_GAMES" in feature_columns
    assert "TEST_AVAILABILITY_AWAY_HAS_PLAYER_EFFECT" in feature_columns


def test_total_prices_are_not_selected_for_historical_rolling_features():
    columns = [
        "total_bet365_price_over",
        "total_bet365_price_under",
        "total_consensus_opener_price_over",
        "spread_bet365_price_home",
        "ml_bet365_price_home",
        "TOTAL_LINE_bet365",
    ]

    assert _get_rolling_price_columns(columns) == [
        "spread_bet365_price_home",
        "ml_bet365_price_home",
    ]


def test_trends_are_only_computed_for_source_columns(monkeypatch):
    source_total_line = "TOTAL_LINE_bet365"
    source_diff = "DIFF_FROM_LINE_bet365"
    trended_parameters = []

    def fake_rolling_stats(df, parameter, **kwargs):
        if parameter not in df.columns:
            return df
        out = df.copy()
        out[f"{parameter}_LAST_ALL_5_MATCHES_BEFORE"] = 0.0
        out[f"{parameter}_SEASON_BEFORE_AVG"] = 0.0
        return out

    def fake_weighted_stats(df, parameter, **kwargs):
        return df

    def fake_season_std(df, param):
        if param not in df.columns:
            return df
        out = df.copy()
        out[f"{param}_SEASON_BEFORE_STD"] = 0.0
        return out

    def fake_trend_slope(df, parameter, **kwargs):
        trended_parameters.append(parameter)
        return df

    monkeypatch.setattr(rolling_module, "compute_rolling_stats", fake_rolling_stats)
    monkeypatch.setattr(
        rolling_module,
        "compute_rolling_weighted_stats",
        fake_weighted_stats,
    )
    monkeypatch.setattr(rolling_module, "compute_season_std", fake_season_std)
    monkeypatch.setattr(rolling_module, "compute_trend_slope", fake_trend_slope)

    df = pd.DataFrame(
        {
            "TEAM_ID": [1],
            "HOME": [1],
            "GAME_DATE": ["2026-01-01"],
            "SEASON_YEAR": [2025],
            source_total_line: [220.5],
            source_diff: [2.0],
        }
    )

    result = compute_all_rolling_statistics(df)

    assert f"{source_total_line}_LAST_ALL_5_MATCHES_BEFORE" in result.columns
    assert f"{source_diff}_SEASON_BEFORE_AVG" in result.columns
    assert source_total_line in trended_parameters
    assert source_diff in trended_parameters
    assert not any(
        "_LAST_" in parameter or "_SEASON_" in parameter or "_WMA_" in parameter
        for parameter in trended_parameters
    )


def test_points_per_40_rolling_excludes_current_game_value():
    assert "PTS_PER_40" in COLS_TO_AVERAGE

    df = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3"],
            "TEAM_ID": ["team-1", "team-1", "team-1"],
            "HOME": [True, False, True],
            "GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05"]),
            "SEASON_YEAR": [2025, 2025, 2025],
            "PTS_PER_40": [80.0, 90.0, 999.0],
        }
    )

    result = compute_rolling_stats(
        df,
        "PTS_PER_40",
        window=5,
        add_extra_season_avg=True,
        group_by_season=False,
    ).set_index("GAME_ID")

    assert result.loc["g3", "PTS_PER_40_LAST_ALL_5_MATCHES_BEFORE"] == pytest.approx(
        85.0
    )


def test_points_per_40_raw_source_is_not_selected_but_lagged_features_are():
    selected = select_training_columns(
        pd.DataFrame(
            {
                "GAME_ID": [1],
                "TOTAL_POINTS": [220.0],
                "PTS_PER_40_TEAM_HOME": [95.0],
                "PTS_PER_40_TEAM_AWAY": [92.0],
                "PTS_PER_40_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME": [94.0],
                "PTS_PER_40_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY": [91.0],
            }
        ),
        original_columns=[],
    )

    assert "PTS_PER_40_TEAM_HOME" not in selected.columns
    assert "PTS_PER_40_TEAM_AWAY" not in selected.columns
    assert "PTS_PER_40_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME" in selected.columns
    assert "PTS_PER_40_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY" in selected.columns
    assert "TOTAL_POINTS" in selected.columns


def test_overtime_history_features_are_selected_for_both_teams():
    overtime_features = {
        f"{feature_name}_TEAM_{side}": [0.0]
        for feature_name in OVERTIME_HISTORY_FEATURE_COLUMNS
        for side in ("HOME", "AWAY")
    }
    selected = select_training_columns(
        pd.DataFrame(
            {
                "GAME_ID": [1],
                "TOTAL_POINTS": [220.0],
                **overtime_features,
            }
        ),
        original_columns=[],
    )

    assert set(overtime_features).issubset(selected.columns)
