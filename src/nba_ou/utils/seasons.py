import pandas as pd


def classify_season_type(game_id):
    if game_id.startswith("001"):
        return "Preseason"
    elif game_id.startswith("002"):
        return "Regular Season"
    elif game_id.startswith("003"):
        return "All Star"
    elif game_id.startswith("004"):
        return "Playoffs"
    elif game_id.startswith("005"):
        return "Playoffs"  # in reality, this should be "Play-In Tournament", but for simplicity we'll use "Playoffs"
    elif game_id.startswith("006"):
        return "In-Season Final Game"
    return "Unknown"

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

    # Determine the season year for the target date
    # NBA season runs from October (month 10) to June
    # If date is Jan-Jun, it's part of season that started previous year
    # If date is Jul-Dec, it's part of season that will start this year (or just ended)
    target_year = date_to_train_until.year
    target_month = date_to_train_until.month

    if target_month <= 6:
        # Jan-Jun: season started previous year
        end_season_year = target_year - 1
    else:
        # Jul-Dec: season starts this year
        end_season_year = target_year

    # Generate all seasons from 2006 to end_season_year
    seasons = []
    for year in range(2006, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons


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

    # Helper function to determine season year from a date
    def get_season_year(date):
        year = date.year
        month = date.month
        # If date is Jan-Jun, season started previous year
        # If date is Jul-Dec, season starts this year
        return year - 1 if month <= 6 else year

    start_season_year = get_season_year(date_from)
    end_season_year = get_season_year(date_to)

    # Generate all seasons between start and end
    seasons = []
    for year in range(start_season_year, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons
