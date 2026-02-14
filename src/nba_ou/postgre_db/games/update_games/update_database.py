"""
NBA Over/Under Predictor - Database Update Script

This script manages the update of the NBA games database by fetching new game data
from the NBA API and merging it with existing data.
"""

from datetime import datetime

import pandas as pd
from nba_ou.fetch_data.fetch_nba_data.fetch_nba_data import fetch_nba_data
from nba_ou.postgre_db.games.update_games.update_database_utils import (
    get_existing_game_ids_from_db,
    get_existing_player_game_ids_from_db,
)
from nba_ou.postgre_db.games.update_games.upload_games_data_to_db import (
    filter_before_upload,
    upload_games_data_to_db,
)
from nba_ou.postgre_db.players.upload_data_players import upload_players_data_to_db
from nba_ou.utils.general_utils import (
    get_nba_season_nullable_from_date,
    get_season_nullable_from_year,
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


def upload_games_to_postgresql(team_df: pd.DataFrame, games_id_to_exclude=None):
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
    success = upload_games_data_to_db(team_df, exclude_game_ids=games_id_to_exclude)
    if success:
        print("✅ Successfully uploaded data to PostgreSQL!")
    else:
        print("❌ No Games data uploaded to PostgreSQL.")

    return success


def upload_players_to_postgresql(
    players_df: pd.DataFrame, players_game_ids_to_exclude=None
):
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
    success = upload_players_data_to_db(
        players_df, exclude_game_ids=players_game_ids_to_exclude
    )
    if success:
        print("✅ Successfully uploaded player data to PostgreSQL!")
    else:
        print("❌ No player data uploaded to PostgreSQL.")
    return success


def update_team_players_database(
    season_year: str = None, games_id_to_exclude: list = None
) -> tuple:
    """Updates team and player databases for a given season.

    Returns:
        tuple: (limit_reached, exclude_game_ids, new_data_found)
            - limit_reached (bool): Whether API rate limit was reached
            - exclude_game_ids (list): List of game IDs that were excluded/invalid
            - new_data_found (bool): Whether any new data (games or players) was found and uploaded
    """
    exclude_game_ids = set(games_id_to_exclude or [])
    new_data_found = False

    if not season_year:
        date = datetime.now()
        # Get Season to Update
        season_nullable = get_nba_season_nullable_from_date(date)
        season_year = season_nullable[:4]  # Extract first 4 digits
    else:
        season_nullable = get_season_nullable_from_year(season_year)
    print(f"Updating season: {season_nullable}")

    # Query existing game IDs from database
    existing_game_ids = get_existing_game_ids_from_db(season_year)

    # Query existing player game IDs from database
    existing_player_game_ids = get_existing_player_game_ids_from_db(season_year)

    # Use the intersection - only skip games that exist in BOTH databases
    # This ensures we only fetch truly new games
    all_existing_game_ids = existing_game_ids.intersection(existing_player_game_ids)

    # extend all_existing_game_ids with games_id_to_exclude to avoid even fetching that data
    if exclude_game_ids:
        all_existing_game_ids.update(exclude_game_ids)

    print(f"Total existing game IDs in both databases: {len(all_existing_game_ids)}")

    # Fetch new data using existing game IDs from database
    team_df, players_df, limit_reached = fetch_nba_data(
        season_nullable,
        existing_game_ids=all_existing_game_ids,
        n_tries=3,
    )

    if team_df is not None:
        team_df, invalid_game_ids = filter_before_upload(team_df, return_invalid=True)
        if len(invalid_game_ids) > 0:
            exclude_game_ids.update(invalid_game_ids)
        # Upload team/game data to PostgreSQL
        if upload_games_to_postgresql(
            team_df, games_id_to_exclude=list(exclude_game_ids)
        ):
            new_data_found = True

    if players_df is not None:
        # Upload player data to PostgreSQL
        if upload_players_to_postgresql(
            players_df, players_game_ids_to_exclude=list(exclude_game_ids)
        ):
            new_data_found = True

    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached, list(exclude_game_ids), new_data_found
