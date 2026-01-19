from datetime import datetime

import pandas as pd


def filter_by_date_range(
    df: pd.DataFrame,
    older_date_to_include: datetime | None,
    most_recent_date_to_include: datetime,
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    Args:
        df (pd.DataFrame): DataFrame with GAME_DATE column
        older_date_to_include (datetime | None): Starting date for filtering. If None, no lower bound
        most_recent_date_to_include (datetime): End date for filtering

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if older_date_to_include is not None:
        df = df[
            (df["GAME_DATE"] >= older_date_to_include)
            & (df["GAME_DATE"] <= most_recent_date_to_include)
        ]
    else:
        df = df[df["GAME_DATE"] <= most_recent_date_to_include]

    return df
