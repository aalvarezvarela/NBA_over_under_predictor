"""
general_utils.py

General utility functions for use across NBA Over/Under Predictor modules.
"""

from datetime import datetime


def get_nba_season_nullable(date):
    """
    Given a date, returns the NBA season string in the format 'YYYY-YY'.
    If the date is before November, it is considered part of the previous season.
    Args:
        date (str or datetime): Date as a string ('YYYY-MM-DD') or datetime object.
    Returns:
        str: NBA season string (e.g., '2024-25')
    """
    # convert it to date if string
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    year = date.year

    # If we are before September, we are still in the previous season
    if date.month < 9:
        season = f"{year-1}-{str(year)[-2:]}"
    else:
        season = f"{year}-{str(year+1)[-2:]}"

    return season
