"""
NBA Over/Under Predictor - Game Database Utilities

This module contains utility functions for fetching and updating NBA game data
from the NBA API, including game statistics and box scores.
"""

import random
import re
import time

import pandas as pd
import requests
from config.constants import SEASON_TYPE_MAP as SEASON_TYPE_MAPPING
from nba_api.library.http import NBAHTTP
from nba_api.stats.endpoints import (
    BoxScoreAdvancedV3,
    BoxScoreTraditionalV3,
    LeagueGameFinder,
)
from postgre_DB.db_config import connect_games_db, connect_players_db
from tqdm import tqdm

from .mapping_v3_v2 import (
    V3_TO_V2_ADVANCED_PLAYER_MAP,
    V3_TO_V2_ADVANCED_TEAM_MAP,
    V3_TO_V2_TRADITIONAL_MAP,
)


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

    # Query distinct game IDs for the season
    query = """
        SELECT DISTINCT game_id 
        FROM nba_games 
        WHERE season_year = %s
    """
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

    # Query distinct game IDs for the season
    query = """
        SELECT DISTINCT game_id 
        FROM nba_players 
        WHERE season_year = %s
    """
    cursor.execute(query, (int(season_year),))
    game_ids = {row[0] for row in cursor.fetchall()}

    cursor.close()
    if close_conn:
        db_connection.close()

    print(
        f"Found {len(game_ids)} existing player games in database for season {season_year}"
    )
    return game_ids


def reset_nba_http_session():
    """Resets the NBA API HTTP session to prevent stale connections."""
    old_session = NBAHTTP.get_session()
    if old_session is not None:
        old_session.close()
    NBAHTTP._session = None
    NBAHTTP.set_session(requests.Session())


def classify_season_type(game_id: str) -> str:
    """Determines the season type based on the game ID prefix."""
    return SEASON_TYPE_MAPPING.get(game_id[:3], "Unknown")


def fetch_box_score_data(game_id: str, n_tries: int = 3):
    """Fetches traditional and advanced box score data with retries."""
    limit_reached = False
    box_score_traditional = None
    box_score_advanced = None

    for api_call in [BoxScoreTraditionalV3, BoxScoreAdvancedV3]:
        attempts = 0
        time.sleep(random.uniform(0.01, 0.03))  # Avoid rate limiting

        while attempts < n_tries:
            try:
                data = api_call(game_id=game_id)

                if api_call == BoxScoreTraditionalV3:
                    box_score_traditional = data
                elif api_call == BoxScoreAdvancedV3:
                    box_score_advanced = data

                break  # Exit the retry loop for this API call if successful

            except Exception as e:
                attempts += 1
                print(
                    f"Attempt {attempts} failed for {game_id} ({api_call.__name__}). Error: {e}"
                )
                reset_nba_http_session()
                if attempts < n_tries:
                    print(f"Retrying in {20 * attempts} seconds...")
                    time.sleep(20 * attempts)
                else:
                    print(
                        f"Failed to fetch data for {game_id} ({api_call.__name__}). Max attempts reached."
                    )
                    limit_reached = True

    if not box_score_traditional or not box_score_advanced:
        print(
            f"Warning: Missing data for game_id {game_id}. Traditional: {box_score_traditional is not None}, Advanced: {box_score_advanced is not None}"
        )

    return box_score_traditional, box_score_advanced, limit_reached


def merge_stats(player_trad, player_adv, team_trad, team_adv, game_id):
    """Merges traditional and advanced stats for both players and teams."""
    try:
        player_stats = pd.merge(
            player_trad,
            player_adv,
            on=["PLAYER_ID", "GAME_ID", "TEAM_ID"],
            suffixes=("", "_drop"),
        )
        player_stats = player_stats.loc[:, ~player_stats.columns.str.endswith("_drop")]
    except Exception as e:
        print(f"Failed to merge player stats for game_id {game_id}: {e}")
        player_stats = pd.DataFrame()

    try:
        team_stats = pd.merge(
            team_trad, team_adv, on=["TEAM_ID", "GAME_ID"], suffixes=("", "_drop")
        )
        team_stats = team_stats.loc[:, ~team_stats.columns.str.endswith("_drop")]
    except Exception as e:
        print(f"Failed to merge team stats for game_id {game_id}: {e}")
        team_stats = pd.DataFrame()

    return player_stats, team_stats


def fetch_nba_data(
    season_nullable: str,
    input_df: pd.DataFrame = None,
    input_player_stats: pd.DataFrame = None,
    existing_game_ids: set = None,
    n_tries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Fetches and processes NBA game data for a specified season.

    This function retrieves game statistics from the NBA API for the given season, processes the data,
    and returns updated game and player statistics. It avoids duplicate entries by checking against
    previously fetched data.

    Args:
        season_nullable (str): The NBA season in the format 'YYYY-YY' (e.g., '2023-24').
        input_df (pd.DataFrame, optional): An existing DataFrame containing previously fetched game data.
                                           If provided, new game data will be appended, avoiding duplicates.
                                           Deprecated - use existing_game_ids instead.
        input_player_stats (pd.DataFrame, optional): An existing DataFrame containing previously fetched
                                                     player statistics. If provided, new player stats
                                                     will be appended, avoiding duplicates.
        existing_game_ids (set, optional): A set of game IDs already present in the database.
                                           If provided, only games not in this set will be fetched.
        n_tries (int, optional): The number of attempts to fetch each game's box score data in case of failures.
                                 Default is 3.

    Returns:
        tuple:
            - pd.DataFrame: Updated game data including team statistics.
            - pd.DataFrame: Updated player statistics.
            - bool: Flag indicating if the rate limit was reached during data fetching.

    Raises:
        ValueError: If the `season_nullable` format is incorrect.

    Notes:
        - The function avoids redundant requests by skipping games already present in `existing_game_ids` or `input_df`.
        - The function introduces a delay between API requests to prevent hitting rate limits.
        - If the rate limit is reached (299 games fetched), the function pauses for 5 seconds
          and resets the HTTP session before stopping further requests.
    """

    limit_reached = False
    if not re.match(r"^\d{4}-\d{2}$", season_nullable):
        raise ValueError("Invalid season format. Expected format: 'YYYY-YY'")

    game_finder = LeagueGameFinder(
        season_nullable=season_nullable, league_id_nullable="00"
    )
    games = game_finder.get_data_frames()[0]
    games["SEASON_TYPE"] = games["GAME_ID"].apply(classify_season_type)
    games["HOME"] = games["MATCHUP"].str.contains("vs.")

    # Determine existing game IDs from provided set or input_df (backward compatibility)
    if existing_game_ids is not None:
        existing_ids = existing_game_ids

    else:
        existing_ids = set()

    game_ids = set(games["GAME_ID"]) - existing_ids

    if not game_ids:
        print(
            f"No new games found for season {season_nullable}. Skipping data fetch..."
        )
        return None, None, limit_reached

    all_player_stats, all_team_stats = [], []
    fetched_counter = 0

    for game_id in tqdm(game_ids, desc="Fetching NBA Game Data"):
        time.sleep(random.uniform(0.1, 0.5))  # Avoid rate limiting

        box_score_traditional, box_score_advanced, limit_reached = fetch_box_score_data(
            game_id, n_tries
        )

        if not box_score_traditional or not box_score_advanced:
            continue

        player_trad = box_score_traditional.get_data_frames()[0].fillna("")
        team_trad = box_score_traditional.get_data_frames()[2].fillna("")
        player_adv = box_score_advanced.get_data_frames()[0].fillna("")
        team_adv = box_score_advanced.get_data_frames()[1].fillna("")

        # Apply V3 to V2 mappings to maintain consistency with historical data
        player_trad.rename(columns=V3_TO_V2_TRADITIONAL_MAP, inplace=True)
        team_trad.rename(columns=V3_TO_V2_TRADITIONAL_MAP, inplace=True)
        player_adv.rename(columns=V3_TO_V2_ADVANCED_PLAYER_MAP, inplace=True)
        team_adv.rename(columns=V3_TO_V2_ADVANCED_TEAM_MAP, inplace=True)

        player_stats, team_stats = merge_stats(
            player_trad, player_adv, team_trad, team_adv, game_id
        )
        all_player_stats.append(player_stats)
        all_team_stats.append(team_stats)

        fetched_counter += 1
        if fetched_counter == 299:
            print("Waiting 5 secs to avoid rate limit...")
            time.sleep(5)
            reset_nba_http_session()
            limit_reached = True

        if limit_reached:
            break

    player_stats_df = (
        pd.concat(all_player_stats, ignore_index=True)
        if all_player_stats
        else pd.DataFrame()
    )
    team_stats_df = (
        pd.concat(all_team_stats, ignore_index=True)
        if all_team_stats
        else pd.DataFrame()
    )
    team_stats_df.rename(columns={"TO": "TOV"}, inplace=True)

    # Extract season year from season_nullable (first 4 digits)
    season_year = season_nullable[:4]

    merged_games = pd.merge(
        games, team_stats_df, on=["GAME_ID", "TEAM_ID"], suffixes=("", "_drop")
    )
    merged_games = merged_games.loc[:, ~merged_games.columns.str.endswith("_drop")]
    merged_games["SEASON_YEAR"] = season_year

    if input_df is not None:
        merged_games = pd.concat(
            [input_df, merged_games], ignore_index=True
        ).drop_duplicates(subset=["GAME_ID", "TEAM_ID", "SEASON_ID"], keep="last")
    else:
        # Remove duplicates even if no input_df (safety measure)
        merged_games = merged_games.drop_duplicates(
            subset=["GAME_ID", "TEAM_ID", "SEASON_ID"], keep="last"
        )

    if input_player_stats is not None:
        player_stats_df = pd.concat(
            [input_player_stats, player_stats_df], ignore_index=True
        ).drop_duplicates(subset=["GAME_ID", "PLAYER_ID", "TEAM_ID"], keep="last")
    else:
        # Remove duplicates even if no input_player_stats (safety measure)
        if not player_stats_df.empty:
            player_stats_df = player_stats_df.drop_duplicates(
                subset=["GAME_ID", "PLAYER_ID", "TEAM_ID"], keep="last"
            )

    # Add season_year to player_stats_df if it's not empty
    if not player_stats_df.empty:
        player_stats_df["SEASON_YEAR"] = season_year

    return merged_games, player_stats_df, limit_reached
