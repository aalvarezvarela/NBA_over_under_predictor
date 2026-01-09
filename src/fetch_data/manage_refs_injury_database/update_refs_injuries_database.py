"""
NBA Over/Under Predictor - Referees and Injuries Database Update Script

This script manages the update of the NBA referees and injuries databases by fetching
new data from the NBA API and loading it to PostgreSQL.
"""

import os
import sys
from datetime import datetime

sys.path.append("/home/adrian_alvarez/Projects/NBA_over_under_predictor/src")
import pandas as pd
from postgre_DB.create_injuries_db import load_injuries_to_db
from postgre_DB.create_refs_db import load_refs_to_db
from utils.general_utils import get_nba_season_nullable

from .update_refs_injuries_database_utils import (
    fetch_refs_injuries_data,
    get_existing_injury_game_ids_from_db,
    get_existing_ref_game_ids_from_db,
)


def load_existing_data(filepath: str, dtype: dict):
    """Attempts to load existing CSV data, returns None if file not found."""
    try:
        df = pd.read_csv(filepath, dtype=dtype)
        print(f"Existing data found: {filepath}")
        return df
    except FileNotFoundError:
        print(f"No existing data found: {filepath}. Starting from scratch...")
        return None


def upload_refs_to_postgresql(refs_df: pd.DataFrame):
    """Uploads referee data to PostgreSQL database.

    Args:
        refs_df: DataFrame containing referee data

    Returns:
        bool: True if successful, False otherwise
    """
    if refs_df is None or refs_df.empty:
        print("No referee data to upload to PostgreSQL.")
        return False

    print("\nUploading referee data to PostgreSQL database...")
    success = load_refs_to_db(refs_df, if_exists="append")
    if success:
        print("✅ Successfully uploaded referee data to PostgreSQL!")
    else:
        print("❌ Failed to upload referee data to PostgreSQL.")
    return success


def upload_injuries_to_postgresql(injuries_df: pd.DataFrame):
    """Uploads injury data to PostgreSQL database.

    Args:
        injuries_df: DataFrame containing injury data

    Returns:
        bool: True if successful, False otherwise
    """
    if injuries_df is None or injuries_df.empty:
        print("No injury data to upload to PostgreSQL.")
        return False

    print("\nUploading injury data to PostgreSQL database...")
    success = load_injuries_to_db(injuries_df, if_exists="append")
    if success:
        print("✅ Successfully uploaded injury data to PostgreSQL!")
    else:
        print("❌ Failed to upload injury data to PostgreSQL.")
    return success


def update_refs_injuries_database(
    injury_folder: str = None,
    ref_folder: str = None,
    date=None,
    save_csv: bool = True,
):
    """
    Main function to update referees and injuries databases.

    Args:
        injury_folder: Path to folder where injury CSV backups will be saved
        ref_folder: Path to folder where referee CSV backups will be saved
        date: Date to determine the season (default: today)
        save_csv: Whether to save CSV backups (default: True)

    Returns:
        bool: True if rate limit was reached, False otherwise
    """
    # Clean up modules to avoid stale connections
    remove_modules = [
        module
        for module in sys.modules
        if module.startswith(
            (
                "requests",
                "urllib",
                "urllib3",
                "chardet",
                "charset_normalizer",
                "idna",
                "certifi",
                "http",
                "socket",
                "json",
                "ssl",
                "h11",
                "h2",
                "hpack",
                "brotli",
                "zlib",
                "nba",
            )
        )
    ]

    for module in remove_modules:
        del sys.modules[module]

    # Remove update_refs_injuries_database_utils module if it exists
    database_utils_module = [
        module
        for module in sys.modules
        if "update_refs_injuries_database_utils" in module
    ]

    if database_utils_module:
        for module in database_utils_module:
            print(f"Reloading module: {module}")
            del sys.modules[module]

        from .update_refs_injuries_database_utils import (
            fetch_refs_injuries_data,
            get_existing_injury_game_ids_from_db,
            get_existing_ref_game_ids_from_db,
        )

    if not date:
        date = datetime.now()

    # Get Season to Update
    season_nullable = get_nba_season_nullable(date)
    season_year = season_nullable[:4]  # Extract first 4 digits

    print(f"Updating referees and injuries for season: {season_nullable}")

    # Query existing game IDs from both databases
    existing_injury_game_ids = get_existing_injury_game_ids_from_db(season_year)
    existing_ref_game_ids = get_existing_ref_game_ids_from_db(season_year)

    # Use the intersection of both sets to only skip games that exist in BOTH databases
    all_existing_game_ids = existing_injury_game_ids.intersection(existing_ref_game_ids)

    print(
        f"Total existing game IDs across both databases: {len(all_existing_game_ids)}"
    )

    # Fetch new data using existing game IDs from database
    refs_df, injuries_df, limit_reached = fetch_refs_injuries_data(
        season_nullable,
        existing_game_ids=all_existing_game_ids,
        n_tries=3,
    )

    # Upload to PostgreSQL
    if refs_df is not None and not refs_df.empty:
        upload_refs_to_postgresql(refs_df)

    if injuries_df is not None and not injuries_df.empty:
        upload_injuries_to_postgresql(injuries_df)

    # Save CSV backups if requested
    if save_csv:
        if refs_df is not None and not refs_df.empty and ref_folder:
            os.makedirs(ref_folder, exist_ok=True)
            refs_filename = f"nba_refs_{season_nullable.replace('-', '_')}.csv"
            refs_path = os.path.join(ref_folder, refs_filename)

            # Load existing CSV data
            df_refs_existing = load_existing_data(refs_path, dtype={"GAME_ID": str})

            save_refs_csv_df = (
                pd.concat([df_refs_existing, refs_df])
                if df_refs_existing is not None
                else refs_df
            )
            save_refs_csv_df.drop_duplicates(
                subset=["GAME_ID", "OFFICIAL_ID"], inplace=True
            )
            save_refs_csv_df.reset_index(drop=True, inplace=True)
            save_refs_csv_df.to_csv(refs_path, index=False)
            print(f"Referee data saved to {refs_path}")

        if injuries_df is not None and not injuries_df.empty and injury_folder:
            os.makedirs(injury_folder, exist_ok=True)
            injuries_filename = f"nba_injuries_{season_nullable.replace('-', '_')}.csv"
            injuries_path = os.path.join(injury_folder, injuries_filename)

            # Load existing CSV data
            df_injuries_existing = load_existing_data(
                injuries_path, dtype={"GAME_ID": str}
            )

            save_injuries_csv_df = (
                pd.concat([df_injuries_existing, injuries_df])
                if df_injuries_existing is not None
                else injuries_df
            )
            save_injuries_csv_df.drop_duplicates(
                subset=["GAME_ID", "PLAYER_ID"], inplace=True
            )
            save_injuries_csv_df.reset_index(drop=True, inplace=True)
            save_injuries_csv_df.to_csv(injuries_path, index=False)
            print(f"Injury data saved to {injuries_path}")

    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached


if __name__ == "__main__":
    # Define Data Folders
    INJURY_DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/injury_data/"
    )
    REF_DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/ref_data/"
    )

    update_refs_injuries_database(
        injury_folder=INJURY_DATA_FOLDER, ref_folder=REF_DATA_FOLDER
    )
