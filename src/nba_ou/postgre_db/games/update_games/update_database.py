"""
NBA Over/Under Predictor - Database Update Script

This script manages the update of the NBA games database by fetching new game data
from the NBA API and merging it with existing data.
"""

import os
import sys
from datetime import datetime

import pandas as pd
from nba_ou.postgre_db.games.create_nba_games_db import load_games_data_to_db
from postgre_DB.create_nba_players_db import load_players_data_to_db
from utils.general_utils import get_nba_season_nullable

from .update_database_utils import (
    fetch_nba_data,
    get_existing_game_ids_from_db,
    get_existing_player_game_ids_from_db,
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


def upload_games_to_postgresql(team_df: pd.DataFrame):
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
    success = load_games_data_to_db(team_df)
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
    success = load_players_data_to_db(players_df)
    if success:
        print("✅ Successfully uploaded player data to PostgreSQL!")
    else:
        print("❌ Failed to upload player data to PostgreSQL.")
    return success


def update_database(database_folder: str, date=None, save_csv: bool = True):
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
    database_utils_module = [
        module
        for module in sys.modules
        if "update_database_utils" in module or "fetch_box_score_data" in module
    ]

    if database_utils_module:
        for module in database_utils_module:
            print(f"Reloading module: {module}")
            del sys.modules[module]

        from .update_database_utils import (
            fetch_nba_data,
            get_existing_game_ids_from_db,
            get_existing_player_game_ids_from_db,
        )

    if not date:
        date = datetime.now()
    # Get Season to Update
    season_nullable = get_nba_season_nullable(date)
    season_year = season_nullable[:4]  # Extract first 4 digits

    print(f"Updating season: {season_nullable}")

    # Query existing game IDs from database
    existing_game_ids = get_existing_game_ids_from_db(season_year)

    # Query existing player game IDs from database
    existing_player_game_ids = get_existing_player_game_ids_from_db(season_year)

    # Use the intersection - only skip games that exist in BOTH databases
    # This ensures we only fetch truly new games
    all_existing_game_ids = existing_game_ids.intersection(existing_player_game_ids)

    print(
        f"Total existing game IDs in both databases: {len(all_existing_game_ids)}"
    )

    # Fetch new data using existing game IDs from database
    team_df, players_df, limit_reached = fetch_nba_data(
        season_nullable,
        existing_game_ids=all_existing_game_ids,
        n_tries=3,
    )

    # Save updated data to CSV
    if team_df is not None:
        # Upload team/game data to PostgreSQL
        upload_games_to_postgresql(team_df)

    if players_df is not None:
        # Upload player data to PostgreSQL
        upload_players_to_postgresql(players_df)

    if save_csv:
        os.makedirs(database_folder, exist_ok=True)
        if team_df is not None:
            teams_filename = f"nba_games_{season_nullable.replace('-', '_')}.csv"

            # load existing csv data
            teams_path = os.path.join(database_folder, teams_filename)
            df_teams_existing = load_existing_data(teams_path, dtype={"GAME_ID": str})
            save_csv_df = (
                pd.concat([df_teams_existing, team_df])
                if df_teams_existing is not None
                else team_df
            )
            save_csv_df.drop_duplicates(subset=["GAME_ID"], inplace=True)
            save_csv_df.reset_index(drop=True, inplace=True)
            save_csv_df.to_csv(teams_path, index=False)
            print(f"Data saved to {teams_path}")

        if players_df is not None:
            players_filename = f"nba_players_{season_nullable.replace('-', '_')}.csv"
            players_path = os.path.join(database_folder, players_filename)
            df_players_existing = load_existing_data(
                players_path, dtype={"GAME_ID": str}
            )

            save_players_csv_df = (
                pd.concat([df_players_existing, players_df])
                if df_players_existing is not None
                else players_df
            )
            save_players_csv_df.drop_duplicates(
                subset=["GAME_ID", "PLAYER_ID"], inplace=True
            )
            save_players_csv_df.reset_index(drop=True, inplace=True)
            save_players_csv_df.to_csv(players_path, index=False)
            print(f"Player data saved to {players_path}")

    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached


if __name__ == "__main__":
    # Define Data Folder
    DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data/"
    )
    update_database(DATA_FOLDER)
