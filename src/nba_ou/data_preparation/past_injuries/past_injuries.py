from collections import defaultdict

import numpy as np
import pandas as pd

# Constants for top player statistics
N_TOP_PLAYERS_NON_INJURED = 8
N_TOP_PLAYERS_INJURED = 6


def get_injured_players_dict(df_injuries, df_players=None):
    """
    Build a dictionary: injured_dict[game_id][team_id] -> list of injured players for that game/team.

    Injured players are collected from:
    - `df_injuries` (inactive/injury feed)
    - `df_players` comments when injury wording appears (e.g. "DND - Injury/Illness")

    Args:
        df_injuries (pd.DataFrame): Injury data with GAME_ID, TEAM_ID, PLAYER_ID
        df_players (pd.DataFrame, optional): Player boxscore data with GAME_ID, TEAM_ID,
            PLAYER_ID and COMMENT/COMMENTS column

    Returns:
        dict: Nested dictionary mapping game_id -> team_id -> list of injured player_ids
    """
    injured_dict = defaultdict(lambda: defaultdict(set))

    # Source 1: official injuries table
    if (
        df_injuries is not None
        and not df_injuries.empty
        and {"GAME_ID", "TEAM_ID", "PLAYER_ID"}.issubset(df_injuries.columns)
    ):
        valid_injuries = df_injuries.loc[
            df_injuries["GAME_ID"].notna()
            & df_injuries["TEAM_ID"].notna()
            & df_injuries["PLAYER_ID"].notna(),
            ["GAME_ID", "TEAM_ID", "PLAYER_ID"],
        ].drop_duplicates()

        for game_id, team_id, player_id in valid_injuries.itertuples(index=False):
            injured_dict[game_id][team_id].add(player_id)

    # Source 2: player comment field includes injury text
    if (
        df_players is not None
        and not df_players.empty
        and {"GAME_ID", "TEAM_ID", "PLAYER_ID"}.issubset(df_players.columns)
    ):
        comment_col = None
        for candidate in ["COMMENT", "COMMENTS", "comment", "comments"]:
            if candidate in df_players.columns:
                comment_col = candidate
                break

        if comment_col is not None:
            injury_mask = (
                df_players[comment_col]
                .fillna("")
                .astype(str)
                .str.contains(r"injur|injry", case=False, regex=True)
            )
            valid_comment_injuries = df_players.loc[
                injury_mask
                & df_players["GAME_ID"].notna()
                & df_players["TEAM_ID"].notna()
                & df_players["PLAYER_ID"].notna(),
                ["GAME_ID", "TEAM_ID", "PLAYER_ID"],
            ].drop_duplicates()

            for game_id, team_id, player_id in valid_comment_injuries.itertuples(
                index=False
            ):
                injured_dict[game_id][team_id].add(player_id)

    injured_dict = {
        game_id: {team_id: list(player_ids) for team_id, player_ids in team_map.items()}
        for game_id, team_map in injured_dict.items()
    }

    return injured_dict


def create_player_lookup(df_players):
    """
    Precompute all necessary indexes for fast player lookups.
    Returns a function that can be called with (season_id, team_id, date_to_filter)
    to get the same result as get_players_for_team_in_season but much faster.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame (already sorted by PLAYER_ID, GAME_DATE)

    Returns:
        callable: A lookup function with signature (season_id, team_id, date_to_filter) -> pd.DataFrame
    """
    # Ensure GAME_DATE is datetime
    df_players = df_players.copy()
    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], errors="coerce")

    # df_valid is for RETURNING data (MIN > 0 and non-null PTS)
    df_valid = df_players[(df_players["MIN"] > 0) & (df_players["PTS"].notna())].copy()

    # Sort for season_team_groups indexing
    df_valid.sort_values(
        ["SEASON_ID", "TEAM_ID", "PLAYER_ID", "GAME_DATE"], inplace=True
    )

    # Build index: (season_id, team_id) -> DataFrame slice (for returning data)
    season_team_groups = {}
    for (season_id, team_id), group_df in df_valid.groupby(["SEASON_ID", "TEAM_ID"]):
        season_team_groups[(season_id, team_id)] = group_df

    # For tracking player movements, use ALL data (including MIN=0 games)
    # because a player's last game might be a DNP (MIN=0) but they're still on the team
    df_all = df_players.copy()
    df_all.sort_values(["SEASON_ID", "PLAYER_ID", "GAME_DATE"], inplace=True)

    # Pre-compute: for each (season, player), build sorted list of (date, team_id)
    # This allows finding the last team before a given date
    # IMPORTANT: Use ALL games (not just MIN>0) to track team membership
    player_timeline = defaultdict(
        list
    )  # (season_id, player_id) -> [(date, team_id), ...]
    for (season_id, player_id), grp in df_all.groupby(["SEASON_ID", "PLAYER_ID"]):
        # Sorted by date (chronologically across all teams)
        dates = grp["GAME_DATE"].values
        teams = grp["TEAM_ID"].values
        player_timeline[(season_id, player_id)] = list(zip(dates, teams))

    # Get unique players per (season, team) - use ALL data for membership tracking
    players_by_season_team = {}
    for (season_id, team_id), group_df in df_all.groupby(["SEASON_ID", "TEAM_ID"]):
        players_by_season_team[(season_id, team_id)] = set(
            group_df["PLAYER_ID"].unique()
        )

    empty_df = pd.DataFrame(columns=df_players.columns)

    def lookup(season_id, team_id, date_to_filter):
        """
        Fast lookup for players on a team in a season before a given date.
        """
        # Get players who ever played for this team in this season
        candidate_players = players_by_season_team.get((season_id, team_id))
        if not candidate_players:
            return empty_df

        # Convert date_to_filter to numpy datetime64 for comparison
        date_np = np.datetime64(date_to_filter)

        # Find players whose last game before date_to_filter was with this team
        valid_players = []
        for player_id in candidate_players:
            timeline = player_timeline.get((season_id, player_id), [])
            if not timeline:
                continue

            # Find the last game before date_to_filter
            # timeline is sorted by date
            last_team = None
            for game_date, game_team in timeline:
                if game_date < date_np:
                    last_team = game_team
                else:
                    break

            if last_team == team_id:
                valid_players.append(player_id)

        if not valid_players:
            return empty_df

        # Get the pre-filtered data for this season/team
        df_team_season = season_team_groups.get((season_id, team_id))
        if df_team_season is None or df_team_season.empty:
            return empty_df

        valid_players_set = set(valid_players)

        # Filter by valid players and date
        mask = df_team_season["PLAYER_ID"].isin(valid_players_set) & (
            df_team_season["GAME_DATE"] <= date_to_filter
        )
        result = df_team_season[mask]

        return result

    return lookup


def get_players_for_team_in_season(df_players, season_id, team_id, date_to_filter):
    """
    Returns rows from df_players belonging to (season_id, team_id),
    only for players who had not left by date_to_filter (based on last game).

    NOTE: For batch processing, use create_player_lookup() instead for much better performance.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        season_id (str): Season identifier
        team_id (str): Team identifier
        date_to_filter (datetime): Date to filter by

    Returns:
        pd.DataFrame: Filtered player data for the team in the season
    """
    # Filter same season
    df_season = df_players[df_players["SEASON_ID"] == season_id].copy()
    if df_season.empty:
        return pd.DataFrame(columns=df_players.columns)

    # Only games BEFORE this date
    df_season = df_season[df_season["GAME_DATE"] < date_to_filter]

    # Only consider players who played for this team at least once
    df_with_target_team = df_season[df_season["TEAM_ID"] == team_id]
    if df_with_target_team.empty:
        return pd.DataFrame(columns=df_players.columns)

    # Players who appeared for that team
    possible_ids = set(df_with_target_team["PLAYER_ID"].unique())
    df_season = df_season[df_season["PLAYER_ID"].isin(possible_ids)]

    # Sort by date so we can see each player's last appearance
    df_season.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)

    # For each player, take the last row to see final team
    df_last_game = df_season.groupby("PLAYER_ID", as_index=False).tail(1)
    final_player_ids = df_last_game.loc[
        df_last_game["TEAM_ID"] == team_id, "PLAYER_ID"
    ].unique()

    # Return the relevant rows for these players who truly remain on the team
    df_result = df_players[
        (df_players["SEASON_ID"] == season_id)
        & (df_players["TEAM_ID"] == team_id)
        & (df_players["PLAYER_ID"].isin(final_player_ids))
    ].copy()

    # Filter out players who have not played
    df_result = df_result[df_result["MIN"] > 0]
    df_result = df_result[df_result["GAME_DATE"] <= date_to_filter]

    # Drop rows with NaN points
    df_result = df_result.dropna(subset=["PTS"])

    return df_result
