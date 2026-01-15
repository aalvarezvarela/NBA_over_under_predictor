import numpy as np
import pandas as pd
from nba_ou.data_preparation.historic_games.historic_games_statistics import (
    compute_home_points_conceded_avg,
    compute_trend_slope,
    get_last_5_matchup_excluding_current,
    compute_differences_in_points_conceeded_annotated
)
from tqdm import tqdm


def merge_home_away_and_prepare_training_features(df):
    """
    Merge home and away team data and prepare final training features.

    Args:
        df (pd.DataFrame): Team statistics DataFrame with injury data attached

    Returns:
        pd.DataFrame: Final training-ready DataFrame
    """
    # Rename columns that start with 'TOP' by adding '_BEFORE' at the end
    df.rename(
        columns=lambda x: f"{x}_BEFORE" if x.startswith("TOP") else x, inplace=True
    )

    # Add '_BEFORE' suffix to specified AVG_INJURED_* columns
    avg_injured_cols = [
        "AVG_INJURED_PTS",
        "AVG_INJURED_PACE_PER40",
        "AVG_INJURED_DEF_RATING",
        "AVG_INJURED_OFF_RATING",
        "AVG_INJURED_TS_PCT",
    ]

    df.rename(columns={col: f"{col}_BEFORE" for col in avg_injured_cols}, inplace=True)

    den = df["OFF_RATING_SEASON_BEFORE_AVG"].replace(0, np.nan)
    df["STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE"] = (
        df["TOP1_PLAYER_OFF_RATING_BEFORE"] / den
    )
    df["STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE"] = df[
        "STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE"
    ].fillna(0)

    df["STAR_PTS_PERCENTAGE_BEFORE"] = (
        df["TOP1_PLAYER_PTS_BEFORE"] / df["PTS_SEASON_BEFORE_AVG"]
    )

    # Fill NaNs in injured player columns with zeros
    injured_player_cols = [col for col in df.columns if "INJURED_PLAYER" in col]
    df[injured_player_cols] = df[injured_player_cols].fillna(0)

    # Merge home and away stats
    static_columns = [
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "SEASON_TYPE",
        "SEASON_YEAR",
        "IS_OVERTIME",
    ]

    df_home = df[df["HOME"]].copy().drop(columns="HOME")
    df_away = df[~df["HOME"]].copy().drop(columns="HOME")

    key = static_columns
    if df_home.duplicated(key).any() or df_away.duplicated(key).any():
        raise ValueError(
            "Home/Away tables have duplicate keys; merge would be many-to-many."
        )

    df_merged = pd.merge(
        df_home,
        df_away,
        on=static_columns,
        how="inner",
        suffixes=("_TEAM_HOME", "_TEAM_AWAY"),
    )
    df_merged["TOTAL_POINTS"] = df_merged.PTS_TEAM_HOME + df_merged.PTS_TEAM_AWAY

    # IS_PLAYOFF_GAME based on SEASON_TYPE
    df_merged["IS_PLAYOFF_GAME"] = (
        df_merged["SEASON_TYPE"].str.contains("Playoff", case=False, na=False)
    ).astype(int)

    df_merged["TOTAL_OVER_UNDER_LINE"] = df_merged["TOTAL_OVER_UNDER_LINE_TEAM_HOME"]
    df_merged["SPREAD"] = df_merged["SPREAD_TEAM_HOME"]

    # Apply function
    df_merged = compute_home_points_conceded_avg(df_merged)
    df_merged["DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"] = (
        df_merged["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
        - df_merged["AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME"]
    )
    df_merged["DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"] = (
        df_merged["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        - df_merged["AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME"]
    )

    df_merged = compute_differences_in_points_conceeded_annotated(df_merged)
    df_merged["TEAMS_DIFFERENCE_OVER_UNDER_LINE_BEFORE"] = (
        df_merged["TOTAL_OVER_UNDER_LINE_SEASON_BEFORE_AVG_TEAM_HOME"]
        - df_merged["TOTAL_OVER_UNDER_LINE_SEASON_BEFORE_AVG_TEAM_AWAY"]
    )

    tqdm.pandas()
    # Apply row-by-row, returning a Series of dictionaries
    results_series = df_merged.progress_apply(
        lambda row: get_last_5_matchup_excluding_current(row, df_merged), axis=1
    )

    # Convert that Series of dicts into a DataFrame
    results_df = pd.DataFrame(results_series.tolist(), index=df_merged.index)

    # Finally, concatenate the new columns onto df_merged
    df_merged = pd.concat([df_merged, results_df], axis=1)
    df_merged.sort_values(["TEAM_ID_TEAM_HOME", "GAME_DATE"], ascending=True)

    # Compute trends using linear regression
    df_merged = compute_trend_slope(df_merged, parameter="PTS", window=5)
    df_merged = compute_trend_slope(df_merged, parameter="TS_PCT", window=5)
    df_merged = compute_trend_slope(
        df_merged, parameter="TOTAL_OVER_UNDER_LINE", window=5
    )
    df_merged = df_merged.sort_values(by="GAME_DATE", ascending=False)

    columns = [
        "TEAM_ID",
        "TEAM_CITY",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "MATCHUP",
        "GAME_NUMBER",
    ]

    # Generate new list with _HOME and _AWAY appended
    columns_info_before = [f"{col}_TEAM_HOME" for col in columns] + [
        f"{col}_TEAM_AWAY" for col in columns
    ]

    columns_info_before.extend(
        [
            "SEASON_ID",
            "IS_OVERTIME",
            "GAME_ID",
            "GAME_DATE",
            "SEASON_TYPE",
            "IS_PLAYOFF_GAME",
            "PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY",
            "PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME",
            "SEASON_YEAR",
        ]
    )
    # Insert columns that have BEFORE in the name
    columns_info_before.extend([col for col in df_merged.columns if "BEFORE" in col])

    odds_columns = [
        "TOTAL_OVER_UNDER_LINE",
        "SPREAD",
        "MONEYLINE_TEAM_HOME",
        "MONEYLINE_TEAM_AWAY",
    ]

    columns_info_before.extend(odds_columns)

    df_training = df_merged[columns_info_before + ["TOTAL_POINTS"]].copy()
    df_training = df_training[df_training["SEASON_TYPE"] != "Preseason"]

    df_training["TOTAL_PTS_SEASON_BEFORE_AVG"] = (
        df_training["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        + df_training["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    )

    df_training["TOTAL_PTS_LAST_GAMES_AVG"] = (
        df_training["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        + df_training["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )

    df_training["BACK_TO_BACK"] = (
        df_training["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"] == 1
    ) & (df_training["REST_DAYS_BEFORE_MATCH_TEAM_HOME"] == 1)

    df_training["DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE_MATCH"] = (
        df_training["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        - df_training["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )
    df_training["DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE_MATCH"] = (
        df_training["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
        - df_training["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
    )

    df_training.loc[df_training["MATCHUP_TEAM_HOME"] == 0, "MATCHUP_TEAM_HOME"] = (
        df_training["TEAM_ABBREVIATION_TEAM_HOME"]
        + " vs. "
        + df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
    )

    df_training.loc[df_training["MATCHUP_TEAM_AWAY"] == 0, "MATCHUP_TEAM_AWAY"] = (
        df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
        + " @ "
        + df_training["TEAM_ABBREVIATION_TEAM_HOME"]
    )

    df_training.drop(columns=["IS_OVERTIME"], inplace=True)

    # Move TOTAL_OVER_UNDER_LINE to first column
    first_col = "TOTAL_OVER_UNDER_LINE"
    df_training = df_training[
        [first_col] + [col for col in df_training.columns if col != first_col]
    ]

    return df_training
