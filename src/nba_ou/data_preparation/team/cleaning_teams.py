import pandas as pd


def clean_team_data(df):
    """
    Clean team game data by removing invalid rows.

    This function:
    - Converts GAME_DATE to datetime
    - Drops rows with missing PTS values
    - Removes duplicate game/team entries
    - Filters out rows with 0 minutes or PTS <= 10

    Args:
        df (pd.DataFrame): Team game statistics DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")
    df.dropna(subset=["PTS"], inplace=True)
    df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="first", inplace=True)
    df = df[df["MIN"] != 0]
    df = df[df["PTS"] > 10]
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    return df


def adjust_overtime(df):
    """
    Clean team game data and adjust statistics for overtime games.

    This function:
    - Converts GAME_DATE to datetime and sorts by date
    - Drops rows with missing PTS values
    - Converts TEAM_ID to string
    - Creates IS_OVERTIME flag for games with MIN >= 259
    - Normalizes statistics to 48-minute equivalent for overtime games

    Args:
        df (pd.DataFrame): Team game statistics DataFrame with MIN column in seconds

    Returns:
        pd.DataFrame: Cleaned DataFrame with overtime-adjusted statistics
    """
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    # Handle overtime adjustments
    df["IS_OVERTIME"] = df["MIN"].apply(lambda x: 1 if x >= 259 else 0)
    mask_overtime = df["MIN"] >= 260

    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols_to_adjust = [
        col
        for col in numeric_cols
        if col
        not in ["MIN", "PACE_PER40", "SEASON_ID", "TEAM_ID", "GAME_ID", "IS_OVERTIME"]
    ]
    int_cols = df[cols_to_adjust].select_dtypes(include=["int64"]).columns.tolist()

    df[int_cols] = df[int_cols].astype(float)

    df.loc[mask_overtime, cols_to_adjust] = (
        df.loc[mask_overtime, cols_to_adjust]
        .astype(float)
        .apply(lambda x: x * (240 / df.loc[mask_overtime, "MIN"]), axis=0)
    )
    for col in cols_to_adjust:
        if df[col].dtype == "float64" and col in int_cols:
            df[col] = df[col].round().astype(int)

    df[int_cols] = df[int_cols].round().astype(int)
    print("Overtime adjustments completed.")
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    return df
