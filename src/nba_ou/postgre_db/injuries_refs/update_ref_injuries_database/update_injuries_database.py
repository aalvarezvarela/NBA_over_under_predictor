"""
NBA Over/Under Predictor - Injuries Database Update Script

This script manages the update of the NBA injuries database by fetching
new data from the NBA API and loading it to PostgreSQL.
"""

import os
from datetime import datetime

import pandas as pd
from nba_ou.postgre_db.injuries.creation.create_injuries_db import load_injuries_to_db
from nba_ou.postgre_db.injuries.update_injuries.update_injuries_database_utils import (
    fetch_injuries_data,
    get_existing_injury_game_ids_from_db,
)
from nba_ou.utils.general_utils import get_nba_season_nullable




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


def update_injuries_database(
    date=None,
):
    """
    Main function to update injuries database.

    Args:
        injury_folder: Path to folder where injury CSV backups will be saved
        date: Date to determine the season (default: today)
        save_csv: Whether to save CSV backups (default: True)

    Returns:
        bool: True if rate limit was reached, False otherwise
    """
    if not date:
        date = datetime.now()

    season_nullable = get_nba_season_nullable(date)
    season_year = season_nullable[:4]

    print(f"Updating injuries for season: {season_nullable}")

    existing_injury_game_ids = get_existing_injury_game_ids_from_db(season_year)

    injuries_df, limit_reached = fetch_injuries_data(
        season_nullable,
        existing_game_ids=existing_injury_game_ids,
        n_tries=3,
    )

    if injuries_df is not None and not injuries_df.empty:
        upload_injuries_to_postgresql(injuries_df)

   
    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached


if __name__ == "__main__":
    INJURY_DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/injury_data/"
    )

    update_injuries_database(injury_folder=INJURY_DATA_FOLDER)
