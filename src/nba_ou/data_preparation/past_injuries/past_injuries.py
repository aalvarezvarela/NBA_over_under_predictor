import pandas as pd

# Constants for top player statistics
N_TOP_PLAYERS_NON_INJURED = 8
N_TOP_PLAYERS_INJURED = 6

def get_injured_players_dict(df_injuries):
    """
    Build a dictionary: injured_dict[game_id][team_id] -> list of injured players for that game/team.

    Args:
        df_injuries (pd.DataFrame): Injury data with GAME_ID, TEAM_ID, PLAYER_ID

    Returns:
        dict: Nested dictionary mapping game_id -> team_id -> list of injured player_ids
    """
    injured_dict = {}
    for game_id, df_g in df_injuries.groupby("GAME_ID"):
        team_map = {}
        for t_id, df_t in df_g.groupby("TEAM_ID"):
            team_map[t_id] = df_t["PLAYER_ID"].unique().tolist()
        injured_dict[game_id] = team_map
    return injured_dict


def _get_players_for_team_in_season(df_players, season_id, team_id, date_to_filter):
    """
    Returns rows from df_players belonging to (season_id, team_id),
    only for players who had not left by date_to_filter (based on last game).

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
    df_result = df_result[df_result["GAME_DATE"] < date_to_filter]

    # Drop rows with NaN points
    df_result = df_result.dropna(subset=["PTS"])

    return df_result
