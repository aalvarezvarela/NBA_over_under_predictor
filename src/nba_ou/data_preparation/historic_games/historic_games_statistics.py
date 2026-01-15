import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_differences_in_points_conceeded_annotated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sort_home = ["TEAM_ID_TEAM_HOME", "SEASON_YEAR", "GAME_DATE"]
    sort_away = ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR", "GAME_DATE"]
    if "GAME_ID" in out.columns:
        sort_home.append("GAME_ID")
        sort_away.append("GAME_ID")

    home_col = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_HOME"
    away_col = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_AWAY"

    out.sort_values(sort_home, ascending=True, inplace=True)
    out[home_col] = (
        out.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
            "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    out.sort_values(sort_away, ascending=True, inplace=True)
    out[away_col] = (
        out.groupby(["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"])[
            "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    out.sort_values("GAME_DATE", ascending=False, inplace=True)
    return out

def compute_trend_slope(df, parameter="PTS", window=10):
    """
    Computes the slope of a linear regression line over the last `window` games
    to determine whether a team's performance is increasing, decreasing, or stable.

    Args:
        df (pd.DataFrame): Must contain columns "TEAM_ID_TEAM_HOME", "GAME_DATE", and `param`.
        param (str): The statistic to analyze (e.g., "PTS").
        window (int): Number of last games to consider.

    Returns:
        pd.DataFrame: A modified DataFrame with a new column:
            - f"{param}_TREND_SLOPE_LAST_{window}_GAMES"
    """

    def calculate_slope(series):
        """Applies linear regression to compute the trend slope."""
        if len(series) < 2:
            return 0  # Not enough data for a trend, so we assing 0

        X = np.arange(1, len(series) + 1)  # Time index [1, 2, ..., N]
        Y = series  # Directly use the series

        slope, _, _, _, _ = linregress(X, Y)
        return slope

    for field in ["TEAM_HOME", "TEAM_AWAY"]:
        param = f"{parameter}_{field}"

        trend_col = f"{param}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"

        # Sort games by date in ascending order
        df = df.sort_values([f"TEAM_ID_{field}", "GAME_DATE"], ascending=True)

        # Apply the function per team, shifting by 1 to exclude the current game
        trend_series = (
            df.groupby(f"TEAM_ID_{field}")[param]
            .apply(
                lambda s: s.shift(1)
                .rolling(window, min_periods=2)
                .apply(calculate_slope, raw=True)
            )
            .reset_index(level=0, drop=True)
        )  # Reset index to align with df

        # Assign the result back to the DataFrame
        df[trend_col] = trend_series
    df[f"{parameter}_COMBINED_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"] = (
        df[f"{parameter}_TEAM_HOME_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"]
        + df[f"{parameter}_TEAM_AWAY_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"]
    )
    return df


def get_last_5_matchup_excluding_current(row, df):
    """
    For the given row (which represents one game),
    find the 5 most recent PRIOR matchups between (TEAM_ID_HOME vs TEAM_ID_AWAY).
    Returns a dict with 5 keys: the TOTAL_POINTS in each of those matchups,
    sorted descending by GAME_DATE, excluding the current row's game_date.

    'df' should have columns:
        - TEAM_ID_HOME
        - TEAM_ID_AWAY
        - GAME_DATE
        - PTS_HOME
        - PTS_AWAYdf_merged
    """
    home_team = row["TEAM_ID_TEAM_HOME"]
    away_team = row["TEAM_ID_TEAM_AWAY"]
    current_date = row["GAME_DATE"]

    df_matchups = df[
        (df["TEAM_ID_TEAM_HOME"] == home_team)
        & (df["TEAM_ID_TEAM_AWAY"] == away_team)
        & (df["GAME_DATE"] < current_date)
    ].copy()

    df_matchups.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    df_matchups = df_matchups.head(5)

    totals_list = df_matchups["TOTAL_POINTS"].tolist()

    if len(totals_list) == 0:
        totals_list = [np.nan] * 5
    elif len(totals_list) < 5:
        while len(totals_list) < 5:
            # append the mean of totals_list
            totals_list.append(sum(totals_list) / len(totals_list))

    mean = sum(totals_list) / 5
    return {
        "LAST_1_GAMES_TOTAL_POINTS_BEFORE": totals_list[0],
        "LAST_2_GAMES_TOTAL_POINTS_BEFORE": totals_list[1],
        "LAST_3_GAMES_TOTAL_POINTS_BEFORE": totals_list[2],
        "LAST_4_GAMES_TOTAL_POINTS_BEFORE": totals_list[3],
        "LAST_5_GAMES_TOTAL_POINTS_BEFORE": totals_list[4],
        "LAST_5_GAMES_TOTAL_POINTS_BEFORE_MEAN": mean,
    }


def compute_home_points_conceded_avg(df):
    """
    Computes the average points conceded by the home team when playing at home,
    in all games prior to the current one, within the same season.

    Args:
        df (pd.DataFrame): Must contain columns:
            - TEAM_ID_TEAM_HOME, SEASON_ID, GAME_DATE, PTS_TEAM_AWAY

    Returns:
        pd.DataFrame: The modified DataFrame with a new column.
    """

    # Sort DataFrame so previous games come first
    df = df.sort_values(
        ["TEAM_ID_TEAM_HOME", "SEASON_YEAR", "GAME_DATE"], ascending=True
    )

    # Define the new column name
    col_name = "AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME"

    # Compute rolling average of points conceded at home (excluding current game)
    df[col_name] = df.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
        "PTS_TEAM_AWAY"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Sort again for away calculations
    df = df.sort_values(
        ["SEASON_YEAR", "TEAM_ID_TEAM_AWAY", "GAME_DATE"], ascending=True
    )

    # Compute rolling average of points conceded away by the away team
    df["AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME"] = df.groupby(
        ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"]
    )["PTS_TEAM_HOME"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.sort_values("GAME_DATE", ascending=False)
    return df


def compute_differences_in_points_conceeded_annotated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sort_home = ["TEAM_ID_TEAM_HOME", "SEASON_YEAR", "GAME_DATE"]
    sort_away = ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR", "GAME_DATE"]
    if "GAME_ID" in out.columns:
        sort_home.append("GAME_ID")
        sort_away.append("GAME_ID")

    home_col = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_HOME"
    away_col = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_AWAY"

    out.sort_values(sort_home, ascending=True, inplace=True)
    out[home_col] = (
        out.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
            "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    out.sort_values(sort_away, ascending=True, inplace=True)
    out[away_col] = (
        out.groupby(["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"])[
            "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    out.sort_values("GAME_DATE", ascending=False, inplace=True)
    return out
