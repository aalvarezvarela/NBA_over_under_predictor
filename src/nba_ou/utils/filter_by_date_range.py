from datetime import datetime

import pandas as pd


def filter_by_date_range(
    df: pd.DataFrame, date_from: datetime | None, date_to: datetime
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    Args:
        df (pd.DataFrame): DataFrame with GAME_DATE column
        date_from (datetime | None): Starting date for filtering. If None, no lower bound
        date_to (datetime): End date for filtering

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if date_from is not None:
        df = df[(df["GAME_DATE"] >= date_from) & (df["GAME_DATE"] <= date_to)]
    else:
        df = df[df["GAME_DATE"] <= date_to]

    return df
