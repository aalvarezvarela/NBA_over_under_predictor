"""
NBA Over/Under Predictor - Database Update Script

This script manages the update of the NBA games database by fetching new game data
from the NBA API and merging it with existing data.
"""

import os
import sys
from datetime import datetime

import pandas as pd
from postgre_DB.create_nba_games_db import load_data_to_db
from postgre_DB.create_nba_players_db import load_data_to_db as load_players_to_db

from .update_database_utils import (
    fetch_nba_data,
    get_existing_game_ids_from_db,
    get_existing_player_game_ids_from_db,
    get_nba_season_to_update,
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


def upload_to_postgresql(team_df: pd.DataFrame):
    """Uploads team/game data to PostgreSQL database.

    Args:
        team_df: DataFrame containing game and team statistics

    Returns:
        bool: True if successful, False otherwise
    """

    if team_df is None or team_df.empty:
        print("No data to upload to PostgreSQL.")
        return False

    print("\nUploading data to PostgreSQL database...")
    success = load_data_to_db(team_df)
    if success:
        print("✅ Successfully uploaded data to PostgreSQL!")
    else:
        print("❌ Failed to upload data to PostgreSQL.")
    return success


def upload_players_to_postgresql(players_df: pd.DataFrame):
    """Uploads player data to PostgreSQL database.

    Args:
        players_df: DataFrame containing player statistics

    Returns:
        bool: True if successful, False otherwise
    """
    if players_df is None or players_df.empty:
        print("No player data to upload to PostgreSQL.")
        return False

    print("\nUploading player data to PostgreSQL database...")
    success = load_players_to_db(players_df)
    if success:
        print("✅ Successfully uploaded player data to PostgreSQL!")
    else:
        print("❌ Failed to upload player data to PostgreSQL.")
    return success


def update_database(database_folder: str, date=None):
    # Try to sort the 300 games block issue
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

    # Remove `update_database_utils` module if it exists
    if "update_database_utils" in sys.modules:
        del sys.modules["update_database_utils"]

    if not date:
        date = datetime.now()
    # Get Season to Update
    season_nullable = get_nba_season_to_update(date)
    season_year = season_nullable[:4]  # Extract first 4 digits
    teams_filename = f"nba_games_{season_nullable.replace('-', '_')}.csv"
    players_filename = f"nba_players_{season_nullable.replace('-', '_')}.csv"

    print(f"Updating season: {season_nullable}")

    # Query existing game IDs from database
    existing_game_ids = get_existing_game_ids_from_db(season_year)

    # Query existing player game IDs from database
    existing_player_game_ids = get_existing_player_game_ids_from_db(season_year)

    # Use the union of both sets to avoid fetching games we already have in either DB
    # This ensures we only fetch truly new games
    all_existing_game_ids = existing_game_ids.union(existing_player_game_ids)

    print(
        f"Total existing game IDs across both databases: {len(all_existing_game_ids)}"
    )

    # Fetch new data using existing game IDs from database
    team_df, players_df, limit_reached = fetch_nba_data(
        season_nullable,
        existing_game_ids=all_existing_game_ids,
        n_tries=3,
    )
    os.makedirs(database_folder, exist_ok=True)

    # Save updated data to CSV
    if team_df is not None:
        teams_path = os.path.join(database_folder, teams_filename)
        team_df.to_csv(teams_path, index=False)
        print(f"Data saved to {teams_path}")

        # Upload team/game data to PostgreSQL
        upload_to_postgresql(team_df)

    if players_df is not None:
        players_path = os.path.join(database_folder, players_filename)
        players_df.to_csv(players_path, index=False)
        print(f"Data saved to {players_path}")

        # Upload player data to PostgreSQL
        upload_players_to_postgresql(players_df)

    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached


if __name__ == "__main__":
    # Define Data Folder
    DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data/"
    )
    update_database(DATA_FOLDER)
