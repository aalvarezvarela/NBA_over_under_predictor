"""
NBA Over/Under Predictor - Season Data Merger

This script downloads games and odds data for a given season and merges them
using the existing merge_teams_df_with_odds function.
"""

import configparser
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
from fetch_data.manage_odds_data.update_odds_utils import (
    load_odds_data,
    merge_teams_df_with_odds,
    process_odds_date,
    update_odds_db,
)
from postgre_DB.db_loader import load_games_from_db


def load_config():
    """
    Load configuration from config.ini and config.secrets.ini files.

    Returns:
        tuple: (config, secrets_config) - ConfigParser objects
    """
    # Get the project root directory (go up from src/)
    project_root = current_dir.parent

    # Load main config
    config = configparser.ConfigParser()
    config_path = project_root / "config.ini"
    config.read(config_path)

    # Load secrets config
    secrets_config = configparser.ConfigParser()
    secrets_path = project_root / "config.secrets.ini"
    secrets_config.read(secrets_path)

    return config, secrets_config


def get_odds_api_config():
    """
    Get odds API configuration including headers and base URL.

    Returns:
        tuple: (BASE_URL, HEADERS)
    """
    config, secrets_config = load_config()

    BASE_URL = config.get("Odds", "BASE_URL")
    ODDS_API_KEY = secrets_config.get("Odds", "ODDS_API_KEY")

    HEADERS = {
        "X-RapidAPI-Key": ODDS_API_KEY,
        "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com",
    }

    return BASE_URL, HEADERS


def find_missing_odds_dates(df_merged: pd.DataFrame) -> list:
    """
    Find game dates where TOTAL_OVER_UNDER_LINE is missing.
    Also includes the next day for each missing date to handle UTC timezone differences.

    Args:
        df_merged (pd.DataFrame): Merged dataframe with games and odds

    Returns:
        list: List of unique dates where odds are missing (including next days)
    """
    # Find rows where TOTAL_OVER_UNDER_LINE is NaN/null
    missing_odds_mask = df_merged["TOTAL_OVER_UNDER_LINE"].isna()

    # Get unique dates where odds are missing
    missing_dates = df_merged.loc[missing_odds_mask, "GAME_DATE"].unique()

    # Convert to string format YYYY-MM-DD and collect both original and next day
    missing_dates_str = set()  # Use set to automatically handle duplicates

    for date in missing_dates:
        if pd.isna(date):
            continue

        # Convert to datetime if not already
        if isinstance(date, str):
            date_dt = pd.to_datetime(date)
        else:
            date_dt = pd.to_datetime(date)

        # Add original date
        original_date_str = date_dt.strftime("%Y-%m-%d")
        missing_dates_str.add(original_date_str)

        # Add next day (for UTC timezone differences)
        next_day_dt = date_dt + pd.Timedelta(days=1)
        next_day_str = next_day_dt.strftime("%Y-%m-%d")
        missing_dates_str.add(next_day_str)

    # Filter out dates before October 22 (remove July, August, September, and October 1-21)
    filtered_dates = []
    for date_str in missing_dates_str:
        date_dt = pd.to_datetime(date_str)
        month = date_dt.month
        day = date_dt.day

        # Skip dates from July (7), August (8), September (9), and October 1-21
        if month in [7, 8, 9] or (month == 10 and day < 22):
            continue

        filtered_dates.append(date_str)

    return sorted(filtered_dates)


def fetch_missing_odds(missing_dates: list) -> pd.DataFrame:
    """
    Fetch odds data for missing dates using the odds API.

    Args:
        missing_dates (list): List of dates to fetch odds for

    Returns:
        pd.DataFrame: Combined odds data for all missing dates
    """
    if not missing_dates:
        print("No missing dates found")
        return pd.DataFrame()

    BASE_URL, HEADERS = get_odds_api_config()

    print(f"Fetching odds for {len(missing_dates)} missing dates...")

    all_odds_data = []

    for date in missing_dates:
        print(f"Fetching odds for {date}...")
        try:
            odds_df = process_odds_date(
                date=date, BASE_URL=BASE_URL, HEADERS=HEADERS, is_today=False
            )

            if not odds_df.empty:
                all_odds_data.append(odds_df)
                print(f"  - Found {len(odds_df)} odds records for {date}")
            else:
                print(f"  - No odds found for {date}")

        except Exception as e:
            print(f"  - Error fetching odds for {date}: {e}")
            continue

    if all_odds_data:
        combined_odds = pd.concat(all_odds_data, ignore_index=True)
        print(f"\nTotal fetched odds records: {len(combined_odds)}")
        return combined_odds
    else:
        print("No odds data could be fetched")
        return pd.DataFrame()


def update_missing_odds(season: str, use_metric: str = "most_common") -> pd.DataFrame:
    """
    Download season data, identify missing odds, fetch missing odds, and update database.

    Args:
        season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
        use_metric (str): Either "average" or "most_common" for odds processing

    Returns:
        pd.DataFrame: Updated merged dataframe with new odds
    """
    print(f"Updating missing odds for season: {season}")

    # Get initial merged data
    df_merged = download_and_merge_season_data(season, use_metric)

    # Find missing odds dates
    missing_dates = find_missing_odds_dates(df_merged)

    if not missing_dates:
        print("No missing odds dates found!")
        return False

    print(f"Found {len(missing_dates)} dates with missing odds:")
    for date in missing_dates[:10]:  # Show first 10
        print(f"  - {date}")
    if len(missing_dates) > 10:
        print(f"  ... and {len(missing_dates) - 10} more")

    # Fetch missing odds
    new_odds_df = fetch_missing_odds(missing_dates)

    if not new_odds_df.empty:
        # Update database with new odds
        print("\nUpdating database with new odds...")
        success = update_odds_db(new_odds_df)

        if success:
            print("Successfully updated odds database")
            return True
    else:
        print("No new odds data to update")
        return False


def download_and_merge_season_data(
    season: str, use_metric: str = "most_common"
) -> pd.DataFrame:
    """
    Downloads games and odds data for a given season and merges them.

    Args:
        season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
        use_metric (str): Either "average" or "most_common" for odds processing

    Returns:
        pd.DataFrame: Merged dataframe with games and odds data
    """
    print(f"Processing season: {season}")

    # Extract season year for odds data (first year of season)
    season_year = season.split("-")[0]

    # Download games data
    print("Loading games data...")
    df_games = load_games_from_db(seasons=[season])
    df_games.columns = df_games.columns.str.upper()
    if df_games is None or df_games.empty:
        raise ValueError(f"No games data found for season {season}")

    print(f"Loaded {len(df_games)} games for season {season}")

    # Download odds data
    print("Loading odds data...")
    df_odds = load_odds_data(season_year=season_year)

    if df_odds is None or df_odds.empty:
        raise ValueError(f"No odds data found for season year {season_year}")

    print(f"Loaded {len(df_odds)} odds records for season year {season_year}")

    # Merge the data
    print("Merging games and odds data...")
    df_merged = merge_teams_df_with_odds(df_odds=df_odds, df_team=df_games)

    print(f"Merged dataframe contains {len(df_merged)} records")

    return df_merged


if __name__ == "__main__":
    season = "2022-23"

    success = update_missing_odds(season=season, use_metric="most_common")
    if success:
        print("\nSuccessfully created merged dataframe")
    else:
        print("\nNo updates were made to the merged dataframe")
