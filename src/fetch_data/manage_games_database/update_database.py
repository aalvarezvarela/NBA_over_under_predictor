"""
NBA Over/Under Predictor - Database Update Script

This script manages the update of the NBA games database by fetching new game data
from the NBA API and merging it with existing data.
"""

import os
import sys

import pandas as pd


def load_existing_data(filepath: str, dtype: dict):
    """Attempts to load existing CSV data, returns None if file not found."""
    try:
        df = pd.read_csv(filepath, dtype=dtype)
        print(f"Existing data found: {filepath}")
        return df
    except FileNotFoundError:
        print(f"No existing data found: {filepath}. Starting from scratch...")
        return None


def update_database(database_folder: str, date = None):
    from .update_database_utils import fetch_nba_data, get_nba_season_to_update

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
    if "src.update_database_utils" in sys.modules:
        del sys.modules["src.update_database_utils"]

    if not date:
        from datetime import datetime

        date = datetime.now()
    # Get Season to Update
    season_nullable = get_nba_season_to_update(date)
    teams_filename = f"nba_games_{season_nullable.replace('-', '_')}.csv"
    players_filename = f"nba_players_{season_nullable.replace('-', '_')}.csv"

    print(f"Updating season: {season_nullable}")

    # Load existing team and player data if available
    input_df = load_existing_data(
        os.path.join(database_folder, teams_filename), dtype={"GAME_ID": str}
    )
    input_player_df = load_existing_data(
        os.path.join(database_folder, players_filename), dtype={"GAME_ID": str}
    )

    team_df, players_df, limit_reached = fetch_nba_data(
        season_nullable,
        input_df=input_df,
        input_player_stats=input_player_df,
        n_tries=3,
    )
    os.makedirs(database_folder, exist_ok=True)
    # # Save updated data
    if team_df is not None:
        teams_path = os.path.join(database_folder, teams_filename)
        team_df.to_csv(teams_path, index=False)
        print(f"Data saved to {teams_path}")

    if players_df is not None:
        players_path = os.path.join(database_folder, players_filename)
        players_df.to_csv(players_path, index=False)
        print(f"Data saved to {players_path}")

    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached


if __name__ == "__main__":
    # Define Data Folder
    DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data/"
    )
    update_database(DATA_FOLDER)
