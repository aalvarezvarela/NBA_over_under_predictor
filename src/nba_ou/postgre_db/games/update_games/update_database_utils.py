"""
NBA Over/Under Predictor - Game Database Utilities

This module contains utility functions for fetching and updating NBA game data
from the NBA API, including game statistics and box scores.
"""

from nba_ou.postgre_db.config.db_config import (
    connect_games_db,
    connect_players_db,
    get_schema_name_games,
    get_schema_name_players,
)
from psycopg import sql


def get_existing_game_ids_from_db(season_year: str, db_connection=None) -> set:
    """Query existing game IDs from PostgreSQL database for a specific season.

    Args:
        season_year: The first 4 digits of the season (e.g., '2024')
        db_connection: Optional database connection. If None, attempts to create one.

    Returns:
        set: Set of existing game IDs in the database for the given season
    """

    close_conn = False
    if db_connection is None:
        db_connection = connect_games_db()
        close_conn = True

    cursor = db_connection.cursor()
    game_name = get_schema_name_games()

    # Query distinct game IDs for the season
    query = sql.SQL("""
        SELECT DISTINCT game_id
        FROM {}.{}
        WHERE season_year = %s
        """).format(
        sql.Identifier(game_name),  # schema
        sql.Identifier(game_name),  # table
    )
    cursor.execute(query, (int(season_year),))
    game_ids = {row[0] for row in cursor.fetchall()}

    cursor.close()
    if close_conn:
        db_connection.close()

    print(f"Found {len(game_ids)} existing games in database for season {season_year}")
    return game_ids


def get_existing_player_game_ids_from_db(season_year: str, db_connection=None) -> set:
    """Query existing player game IDs from PostgreSQL database for a specific season.

    Args:
        season_year: The first 4 digits of the season (e.g., '2024')
        db_connection: Optional database connection. If None, attempts to create one.

    Returns:
        set: Set of existing game IDs in the player database for the given season
    """
    # Try to import DB connection function

    close_conn = False
    if db_connection is None:
        try:
            db_connection = connect_players_db()
            close_conn = True
        except Exception as e:
            print(f"Could not connect to players database: {e}")
            return set()

    cursor = db_connection.cursor()
    player_name = get_schema_name_players()
    # Query distinct game IDs for the season
    query = sql.SQL("""
        SELECT DISTINCT game_id 
        FROM {}.{} 
        WHERE season_year = %s
    """).format(
        sql.Identifier(player_name),  # schema
        sql.Identifier(player_name),  # table
    )
    cursor.execute(query, (int(season_year),))
    game_ids = {row[0] for row in cursor.fetchall()}

    cursor.close()
    if close_conn:
        db_connection.close()

    print(
        f"Found {len(game_ids)} existing player games in database for season {season_year}"
    )
    return game_ids
