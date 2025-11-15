import random
import re
import time
from datetime import datetime

import pandas as pd
import requests
from nba_api.library.http import NBAHTTP
from nba_api.stats.endpoints import BoxScoreAdvancedV2, BoxScoreTraditionalV2, LeagueGameFinder
from tqdm import tqdm

# Season Type Mapping
SEASON_TYPE_MAPPING = {
    "001": "Preseason",
    "002": "Regular Season",
    "003": "All Star",
    "004": "Playoffs",
    "005": "Play-In Tournament",
    "006": "In-Season Final Game",
}


def get_nba_season_to_update():
    today = datetime.today()
    year = today.year

    # If we are before July, we are still in the previous season
    if today.month < 11:
        season = f"{year-1}-{str(year)[-2:]}"
    else:
        season = f"{year}-{str(year+1)[-2:]}"

    return season


def reset_nba_http_session():
    """Resets the NBA API HTTP session to prevent stale connections."""
    old_session = NBAHTTP.get_session()
    if old_session is not None:
        old_session.close()
    NBAHTTP._session = None
    NBAHTTP.set_session(requests.Session())
    # reload_nba_api()


# import nba_api  # Explicitly import the main package

# def reload_nba_api():
#     """
#     Reloads NBA API modules to refresh any changes in the imported classes.
#     """
#     importlib.reload(nba_api.library.http)
#     importlib.reload(nba_api.stats.endpoints)

#     # Re-import the necessary classes
#     global NBAHTTP, BoxScoreAdvancedV2, BoxScoreTraditionalV2, LeagueGameFinder
#     from nba_api.library.http import NBAHTTP
#     from nba_api.stats.endpoints import BoxScoreAdvancedV2, BoxScoreTraditionalV2, LeagueGameFinder


def classify_season_type(game_id: str) -> str:
    """Determines the season type based on the game ID prefix."""
    return SEASON_TYPE_MAPPING.get(game_id[:3], "Unknown")


def fetch_box_score_data(game_id: str, n_tries: int = 3):
    """Fetches traditional and advanced box score data with retries."""
    limit_reached = False
    box_score_traditional = None
    box_score_advanced = None

    for api_call in [BoxScoreTraditionalV2, BoxScoreAdvancedV2]:
        attempts = 0
        time.sleep(random.uniform(0.1, 0.3))  # Avoid rate limiting

        while attempts < n_tries:
            try:
                data = api_call(game_id=game_id)

                if api_call == BoxScoreTraditionalV2:
                    box_score_traditional = data
                elif api_call == BoxScoreAdvancedV2:
                    box_score_advanced = data

                break  # Exit the retry loop for this API call if successful

            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed for {game_id} ({api_call.__name__}). Error: {e}")
                reset_nba_http_session()
                if attempts < n_tries:
                    print(f"Retrying in {20 * attempts} seconds...")
                    time.sleep(20 * attempts)
                else:
                    print(f"Failed to fetch data for {game_id} ({api_call.__name__}). Max attempts reached.")
                    limit_reached = True

    if not box_score_traditional or not box_score_advanced:
        print(
            f"Warning: Missing data for game_id {game_id}. Traditional: {box_score_traditional is not None}, Advanced: {box_score_advanced is not None}"
        )

    return box_score_traditional, box_score_advanced, limit_reached


def merge_stats(player_trad, player_adv, team_trad, team_adv, game_id):
    """Merges traditional and advanced stats for both players and teams."""
    try:
        player_stats = pd.merge(player_trad, player_adv, on=["PLAYER_ID", "GAME_ID", "TEAM_ID"], suffixes=("", "_drop"))
        player_stats = player_stats.loc[:, ~player_stats.columns.str.endswith("_drop")]
    except Exception as e:
        print(f"Failed to merge player stats for game_id {game_id}: {e}")
        player_stats = pd.DataFrame()

    try:
        team_stats = pd.merge(team_trad, team_adv, on=["TEAM_ID", "GAME_ID"], suffixes=("", "_drop"))
        team_stats = team_stats.loc[:, ~team_stats.columns.str.endswith("_drop")]
    except Exception as e:
        print(f"Failed to merge team stats for game_id {game_id}: {e}")
        team_stats = pd.DataFrame()

    return player_stats, team_stats


def fetch_nba_data(
    season_nullable: str, input_df: pd.DataFrame = None, input_player_stats: pd.DataFrame = None, n_tries: int = 3
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
        input_player_stats (pd.DataFrame, optional): An existing DataFrame containing previously fetched
                                                     player statistics. If provided, new player stats
                                                     will be appended, avoiding duplicates.
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
        - The function avoids redundant requests by skipping games already present in `input_df`.
        - The function introduces a delay between API requests to prevent hitting rate limits.
        - If the rate limit is reached (299 games fetched), the function pauses for 30 seconds
          and resets the HTTP session before stopping further requests.
    """

    limit_reached = False
    if not re.match(r"^\d{4}-\d{2}$", season_nullable):
        raise ValueError("Invalid season format. Expected format: 'YYYY-YY'")

    game_finder = LeagueGameFinder(season_nullable=season_nullable, league_id_nullable="00")
    games = game_finder.get_data_frames()[0]
    games["SEASON_TYPE"] = games["GAME_ID"].apply(classify_season_type)
    games["HOME"] = games["MATCHUP"].str.contains("vs.")

    existing_game_ids = set(input_df["GAME_ID"]) if input_df is not None else set()
    game_ids = set(games["GAME_ID"]) - existing_game_ids

    if not game_ids:
        print(f"No new games found for season {season_nullable}. Skipping data fetch...")
        return None, None, limit_reached

    all_player_stats, all_team_stats = [], []
    fetched_counter = 0

    for game_id in tqdm(game_ids, desc="Fetching NBA Game Data"):
        time.sleep(random.uniform(0.5, 1.0))  # Avoid rate limiting

        box_score_traditional, box_score_advanced, limit_reached = fetch_box_score_data(game_id, n_tries)

        if not box_score_traditional or not box_score_advanced:
            continue

        player_trad = box_score_traditional.get_data_frames()[0].fillna("")
        team_trad = box_score_traditional.get_data_frames()[1].fillna("")
        player_adv = box_score_advanced.get_data_frames()[0].fillna("")
        team_adv = box_score_advanced.get_data_frames()[1].fillna("")

        player_stats, team_stats = merge_stats(player_trad, player_adv, team_trad, team_adv, game_id)
        all_player_stats.append(player_stats)
        all_team_stats.append(team_stats)

        fetched_counter += 1
        if fetched_counter == 299:
            print("Waiting 30 secs to avoid rate limit...")
            time.sleep(30)
            reset_nba_http_session()
            limit_reached = True
        
        if limit_reached:
            break

    player_stats_df = pd.concat(all_player_stats, ignore_index=True) if all_player_stats else pd.DataFrame()
    team_stats_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()
    team_stats_df.rename(columns={"TO": "TOV"}, inplace=True)

    merged_games = pd.merge(games, team_stats_df, on=["GAME_ID", "TEAM_ID"], suffixes=("", "_drop"))
    merged_games = merged_games.loc[:, ~merged_games.columns.str.endswith("_drop")]

    if input_df is not None:
        merged_games = pd.concat([input_df, merged_games], ignore_index=True).drop_duplicates()

    if input_player_stats is not None:
        player_stats_df = pd.concat([input_player_stats, player_stats_df], ignore_index=True).drop_duplicates()

    return merged_games, player_stats_df, limit_reached
