"""
general_utils.py

General utility functions for use across NBA Over/Under Predictor modules.
"""

from datetime import datetime


def _with_before_suffix(name: str) -> str:
    return name if name.endswith("_BEFORE") else f"{name}_BEFORE"


def get_season_year_from_date(date: datetime | str) -> int:
    """
    Given a date, returns the starting year of the NBA season it belongs to.

    July remains in the season that started in the previous calendar year.
    August begins the next season bucket. The delayed 2020-21 season is handled
    as a special case, with dates through October 2020 assigned to 2019-20.

    Args:
        date (datetime or str): The date to evaluate as datetime object or string ('YYYY-MM-DD').
    Returns:
        int: The starting year of the NBA season.
    """
    # Convert to datetime if string
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    year = date.year
    month = date.month

    month_limit = 10 if year == 2020 else 7
    if month <= month_limit:
        return year - 1
    return year


def get_season_nullable_from_year(season_year: int | str) -> str:
    """
    Given a season start year (e.g., 2024), returns the NBA season string in the format 'YYYY-YY'.

    Args:
        season_year (int or str): The starting year of the NBA season.
    Returns:
        str: NBA season string (e.g., '2024-25')
    """
    season_year = int(season_year)
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def get_nba_season_nullable_from_date(date):
    """
    Given a date, returns the NBA season string in the format 'YYYY-YY',
    using the canonical season-year boundary from
    :func:`get_season_year_from_date`.
    Args:
        date (str or datetime): Date as a string ('YYYY-MM-DD') or datetime object.
    Returns:
        str: NBA season string (e.g., '2024-25')
    """
    return get_season_nullable_from_year(get_season_year_from_date(date))
