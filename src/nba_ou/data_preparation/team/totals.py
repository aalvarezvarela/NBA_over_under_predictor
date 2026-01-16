import pandas as pd


def compute_total_points_features(df):
    """
    Compute total points and related features.

    This function:
    - Computes TOTAL_POINTS as sum of PTS per game
    - Computes DIFF_FROM_LINE (actual - line)
    - Ensures GAME_DATE is datetime
    - Adds SEASON_TYPE and SEASON_YEAR columns

    Args:
        df (pd.DataFrame): Team game statistics DataFrame with odds merged

    Returns:
        pd.DataFrame: DataFrame with computed features
    """
    df["TOTAL_POINTS"] = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["DIFF_FROM_LINE"] = df["TOTAL_POINTS"] - df["TOTAL_OVER_UNDER_LINE"]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")

    return df
