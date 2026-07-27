import pandas as pd

from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.utils.general_utils import get_season_year_from_date


def classify_season_type(game_id: str) -> str:
    """Return the canonical season type for an NBA game ID."""
    return SEASON_TYPE_MAP.get(game_id[:3], "Unknown")


def get_all_seasons_from_2006(date_to_train_until):
    """
    Get all NBA seasons from 2006-07 until the season containing date_to_train_until.

    Args:
        date_to_train_until (datetime): The target date

    Returns:
        list: List of season strings in format "YYYY-YY" (e.g., ["2006-07", "2007-08", ...])
    """
    if isinstance(date_to_train_until, str):
        date_to_train_until = pd.to_datetime(date_to_train_until)

    end_season_year = get_season_year_from_date(date_to_train_until)

    # Generate all seasons from 2006 to end_season_year
    seasons = []
    for year in range(2006, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons


def get_season_start_date_n_seasons_back(
    n_seasons: int, reference_date=None
) -> pd.Timestamp:
    """
    Get the start date of the season N seasons back from reference_date.

    NBA seasons start in October. For example, if n_seasons=2 and we're in
    the 2025-26 season, returns the start of the 2024-25 season (Oct 2024).

    Args:
        n_seasons (int): Number of seasons to include (1 = current season only,
                        2 = current + previous, etc.)
        reference_date (datetime, optional): Reference date to compute from.
                                             Defaults to today.

    Returns:
        pd.Timestamp: Start date (October 1st) of the earliest season to include.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    if isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)

    current_season_start_year = get_season_year_from_date(reference_date)

    # Go back n_seasons - 1 (since n_seasons=1 means current season only)
    target_season_start_year = current_season_start_year - (n_seasons - 1)

    return pd.Timestamp(year=target_season_start_year, month=10, day=1)


def get_seasons_between_dates(date_from, date_to):
    """
    Get all NBA seasons between two dates (inclusive).

    Args:
        date_from (datetime or str): The start date
        date_to (datetime or str): The end date

    Returns:
        list: List of season strings in format "YYYY-YY" (e.g., ["2006-07", "2007-08", ...])
    """
    if isinstance(date_from, str):
        date_from = pd.to_datetime(date_from)
    if isinstance(date_to, str):
        date_to = pd.to_datetime(date_to)

    start_season_year = get_season_year_from_date(date_from)
    end_season_year = get_season_year_from_date(date_to)

    # Generate all seasons between start and end
    seasons = []
    for year in range(start_season_year, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons
