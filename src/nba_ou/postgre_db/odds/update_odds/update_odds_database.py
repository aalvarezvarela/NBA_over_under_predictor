"""
NBA Over/Under Predictor - Odds Database Update

This module provides functions to update the odds database with missing odds data.
It combines season-wide updates with game-by-game verification.
"""

import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.fetch_data.fetch_odds_data.get_odds_date import process_odds_date
from nba_ou.postgre_db.config.db_loader import load_games_from_db
from nba_ou.postgre_db.odds.create.create_nba_odds_db import (
    database_exists,
    schema_exists,
)
from nba_ou.postgre_db.odds.update_odds.odds_no_data_dates import ODDS_NO_DATA_DATES
from nba_ou.postgre_db.odds.update_odds.upload_to_odds_db import (
    get_existing_odds_from_db,
    upload_to_odds_db,
)
from tqdm import tqdm


def update_odds_database(
    season_to_download: str,
    ODDS_API_KEY: str,
    BASE_URL: str,
    check_missing_by_game: bool = True,
    save_pickle: bool = False,
    pickle_path: str = None,
) -> pd.DataFrame:
    """
    Update odds database with missing odds data for a given season.

    This function:
    1. Checks if database and schema exist
    2. Loads existing odds from database
    3. Identifies missing odds dates from season games
    4. Fetches and updates missing odds
    5. Optionally checks game-by-game for missing odds

    Args:
        season_to_download (str): Season in format 'YYYY-YY' (e.g., '2024-25')
        ODDS_API_KEY (str): API key for odds service
        BASE_URL (str): Base URL for odds API
        date_to_predict (str, optional): Date to fetch today/tomorrow odds for. Format: 'YYYY-MM-DD'
        check_missing_by_game (bool): If True, performs game-by-game missing odds check

    Returns:
        pd.DataFrame: Updated odds dataframe with all fetched odds
    """
    # Safety check: verify database exists
    if not database_exists():
        raise RuntimeError(
            "Database does not exist. Please create the database first using create_nba_odds_db.py"
        )

    # Safety check: verify schema exists
    if not schema_exists():
        raise RuntimeError(
            "Odds schema does not exist. Please create the schema first using create_nba_odds_db.py"
        )

    print(f"Processing odds for season: {season_to_download}")

    HEADERS = {
        "x-rapidapi-key": ODDS_API_KEY,
        "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
    }

    # Load existing odds from database (primary source)
    season_year = season_to_download[:4]  # Extract year from season like "2024-25"
    df_odds = get_existing_odds_from_db(season_year)

    # Ensure game_date is datetime
    if not df_odds.empty:
        df_odds["game_date"] = pd.to_datetime(df_odds["game_date"])

    excluded_types = [
        code
        for code, name in SEASON_TYPE_MAP.items()
        if name in ["All Star", "Preseason"]
    ]

    if not check_missing_by_game:
        # Use NBA API to get games and find missing dates
        print("Loading games from NBA API...")
        game_finder = LeagueGameFinder(
            season_nullable=season_to_download, league_id_nullable="00"
        )
        games_df = game_finder.get_data_frames()[0]

        # Filter out All Star and Preseason games based on game ID
        # GAME_ID format: [season_type][season][game_number]

        if "GAME_ID" in games_df.columns:
            games_df = games_df[
                ~games_df["GAME_ID"].astype(str).str[0:3].isin(excluded_types)
            ].copy()
            print(
                f"Filtered to {len(games_df)} games (excluded All Star and Preseason)"
            )

        # Get unique dates from games
        unique_dates = sorted(
            pd.to_datetime(games_df["GAME_DATE"]).dt.strftime("%Y-%m-%d").unique(),
            reverse=True,
        )

        # Filter out dates that already have odds
        if not df_odds.empty:
            existing_dates = df_odds["game_date"].dt.strftime("%Y-%m-%d").values
            unique_dates = [date for date in unique_dates if date not in existing_dates]

        # Filter out off-season dates (August, September, early October)
        unique_dates = [
            date
            for date in unique_dates
            if pd.to_datetime(date).month not in [8, 9]
            and not (pd.to_datetime(date).month == 10 and pd.to_datetime(date).day < 22)
        ]

        # Fetch missing odds by date
        if unique_dates:
            print(f"Found {len(unique_dates)} dates without odds. Fetching...")

            new_odds_data = []
            for date in tqdm(unique_dates, desc="Processing odds per date"):
                df_day = process_odds_date(
                    date,
                    BASE_URL,
                    HEADERS,
                    is_today=False,
                    save_pickle=save_pickle,
                    pickle_path=pickle_path,
                )
                if df_day.empty:
                    print(f"No data for {date}")
                    continue

                # Collect new odds
                new_odds_data.append(df_day)

            # Combine and update database with only the newly fetched odds
            if new_odds_data:
                df_new_odds = pd.concat(new_odds_data, ignore_index=True)
                print(
                    f"Updating database with {len(df_new_odds)} newly fetched odds..."
                )
                upload_to_odds_db(df_new_odds)

            else:
                print("No odds data could be fetched")
        else:
            print("No missing dates found in initial check")
    if check_missing_by_game:
        # Use database to get games and perform game-by-game verification
        print("Loading games from database for game-by-game verification...")
        games_df = load_games_from_db(seasons=[season_to_download])

        if games_df is None or games_df.empty:
            print(
                f"Warning: No games found in database for season {season_to_download}"
            )
            print(f"Total odds records: {len(df_odds)}")
            return df_odds

        # Ensure column names are uppercase for consistency
        games_df.columns = games_df.columns.str.upper()

        if "GAME_ID" in games_df.columns:
            games_df = games_df[
                ~games_df["GAME_ID"].astype(str).str[0:3].isin(excluded_types)
            ].copy()
            print(
                f"Filtered to {len(games_df)} games (excluded All Star and Preseason)"
            )

        print("\nPerforming game-by-game missing odds verification...")
        missing_odds_dates = _find_missing_odds_dates_from_games(games_df, df_odds)

        if missing_odds_dates:
            print(f"Found {len(missing_odds_dates)} dates with missing game odds")
            new_odds = _fetch_missing_odds(
                missing_odds_dates,
                BASE_URL,
                HEADERS,
                save_pickle=save_pickle,
                pickle_path=pickle_path,
            )

            if not new_odds.empty:
                print("Updating database with missing game odds...")
                upload_to_odds_db(new_odds)
        else:
            print("No missing game odds found")

    print(f"\nTotal odds records: {len(df_odds)}")
    return True


def _find_missing_odds_dates_from_games(
    games_df: pd.DataFrame,
    df_odds: pd.DataFrame,
) -> list:
    """
    Find dates where games exist but odds might be missing.

    Args:
        games_df (pd.DataFrame): Games dataframe from NBA API
        df_odds (pd.DataFrame): Current odds dataframe

    Returns:
        list: List of dates (with next day) where odds might be missing
    """
    # Get all game dates
    game_dates = pd.to_datetime(games_df["GAME_DATE"]).dt.strftime("%Y-%m-%d").unique()

    # Get all odds dates
    if df_odds.empty:
        odds_dates = set()
    else:
        odds_dates = set(df_odds["game_date"].dt.strftime("%Y-%m-%d").values)

    # Find dates with potential missing odds (including next day for UTC differences)
    missing_dates = set()
    for date in game_dates:
        date_dt = pd.to_datetime(date)

        # Add date and next day to check
        date_str = date_dt.strftime("%Y-%m-%d")
        if date_str not in odds_dates:
            missing_dates.add(date_str)

        next_day_str = (date_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day_str not in odds_dates:
            missing_dates.add(next_day_str)

    return sorted(list(missing_dates))


def _fetch_missing_odds(
    missing_dates: list,
    BASE_URL: str,
    HEADERS: dict,
    save_pickle: bool = False,
    pickle_path: str = None,
) -> pd.DataFrame:
    """
    Fetch odds data for missing dates using the odds API.

    Args:
        missing_dates (list): List of dates to fetch odds for
        BASE_URL (str): Base URL for odds API
        HEADERS (dict): API headers with authentication
        save_pickle (bool): Whether to save raw matches as pickle
        pickle_path (str): Path to save pickle files

    Returns:
        pd.DataFrame: Combined odds data for all missing dates
    """
    # Import the list of dates to skip

    # Exclude dates with no data
    filtered_dates = [d for d in missing_dates if d not in ODDS_NO_DATA_DATES]
    if not filtered_dates:
        print("No missing dates to fetch after excluding known no-data dates.")
        return pd.DataFrame()

    print(f"Fetching odds for {len(filtered_dates)} potentially missing dates...")

    all_odds_data = []

    for date in tqdm(filtered_dates, desc="Fetching missing odds"):
        odds_df = process_odds_date(
            date=date,
            BASE_URL=BASE_URL,
            HEADERS=HEADERS,
            is_today=False,
            save_pickle=save_pickle,
            pickle_path=pickle_path,
        )

        if not odds_df.empty:
            all_odds_data.append(odds_df)

    if all_odds_data:
        combined_odds = pd.concat(all_odds_data, ignore_index=True)
        print(f"Fetched {len(combined_odds)} new odds records")
        return combined_odds
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    import configparser
    from pathlib import Path

    # Load configuration
    project_root = Path(__file__).parent.parent.parent.parent.parent
    config = configparser.ConfigParser()
    config.read(project_root / "config.ini")

    secrets_config = configparser.ConfigParser()
    secrets_config.read(project_root / "config.secrets.ini")

    BASE_URL = config.get("Odds", "BASE_URL")
    ODDS_API_KEY = secrets_config.get("Odds", "ODDS_API_KEY")
    SAVE_PICKLE = config.getboolean("Odds", "SAVE_ODDS_PICKLE", fallback=False)
    PICKLE_PATH = config.get("Odds", "ODDS_PICKLE_PATH", fallback=None)

    # Update odds for current season
    season = "2024-25"
    date_to_predict = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d")

    df_odds = update_odds_database(
        season_to_download=season,
        ODDS_API_KEY=ODDS_API_KEY,
        BASE_URL=BASE_URL,
        check_missing_by_game=True,
        save_pickle=SAVE_PICKLE,
        pickle_path=PICKLE_PATH,
    )

    print(f"\nFinal odds dataframe shape: {df_odds.shape}")
