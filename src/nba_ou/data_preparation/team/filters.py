from nba_ou.utils.seasons import classify_season_type


def filter_valid_games(df):
    """
    Filter to games with exactly 2 team entries and exclude invalid game types.

    This ensures that only complete games (both teams present) are included
    in the dataset, removing any incomplete or malformed game records.
    Also excludes Preseason and All Star games.

    Args:
        df (pd.DataFrame): Team game statistics DataFrame with SEASON_TYPE column

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid games
    """
    # Filter to games with exactly 2 team entries
    valid_games = df["GAME_ID"].value_counts()
    valid_games = valid_games[valid_games == 2].index

    df = df[df["GAME_ID"].isin(valid_games)]
    df.loc[:, "SEASON_TYPE"] = df["GAME_ID"].apply(classify_season_type)
    df.loc[:, "SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)
    df = df[~df["SEASON_TYPE"].isin(["Preseason", "All Star"])]

    return df
