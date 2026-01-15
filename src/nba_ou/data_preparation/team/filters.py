
def filter_valid_games(df):
    """
    Filter to games with exactly 2 team entries.

    This ensures that only complete games (both teams present) are included
    in the dataset, removing any incomplete or malformed game records.

    Args:
        df (pd.DataFrame): Team game statistics DataFrame

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid games
    """
    valid_games = df["GAME_ID"].value_counts()
    valid_games = valid_games[valid_games == 2].index

    df = df[df["GAME_ID"].isin(valid_games)]

    return df