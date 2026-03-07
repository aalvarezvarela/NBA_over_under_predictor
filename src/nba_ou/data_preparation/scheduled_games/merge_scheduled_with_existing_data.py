import pandas as pd
from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION


def standardize_and_merge_scheduled_games_to_team_data(df, scheduled_games):
    """
    Standardizes the `games` DataFrame to match `df` and merges them.

    - Renames columns to align with `df`
    - Expands `games` to include separate home and away team rows
    - Merges while keeping only relevant columns

    Parameters:
        df (pd.DataFrame): Main DataFrame containing existing game stats.
        games (pd.DataFrame): DataFrame containing new game records.

    Returns:
        pd.DataFrame: Merged and standardized DataFrame.
    """
    # Ensure column names match
    games_renamed = scheduled_games.rename(
        columns={
            "GAME_DATE_EST": "GAME_DATE",
            "HOME_TEAM_ID": "TEAM_ID",
            "VISITOR_TEAM_ID": "TEAM_ID_AWAY",  # Temporarily rename to avoid conflict
        }
    )
    # set string to Team_ID of games
    games_renamed["TEAM_ID"] = games_renamed["TEAM_ID"].astype(str)
    games_renamed["TEAM_ID_AWAY"] = games_renamed["TEAM_ID_AWAY"].astype(str)
    games_renamed["TEAM_ID_AWAY"] = games_renamed["TEAM_ID_AWAY"].astype(str)
    games_renamed["GAME_DATE"] = pd.to_datetime(games_renamed["GAME_DATE"])
    games_renamed["SEASON_YEAR"] = games_renamed["SEASON"].astype(str).str[:4]
    games_renamed["SEASON_YEAR"] = games_renamed["SEASON_YEAR"].astype(str)
    # Recreate SEASON_ID using the first digit before the first '00' in GAME_ID and SEASON_YEAR
    games_renamed["SEASON_PREFIX"] = games_renamed["GAME_ID"].astype(str).str[2]
    games_renamed["SEASON_ID"] = (
        games_renamed["SEASON_PREFIX"] + games_renamed["SEASON_YEAR"]
    )
    games_renamed["SEASON_ID"] = games_renamed["SEASON_ID"].astype(str)
    # Create separate DataFrames for home and away teams
    cols_to_keep = ["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON_ID"]
    if "GAME_TIME" in games_renamed.columns:
        cols_to_keep.append("GAME_TIME")

    home_games = games_renamed[cols_to_keep].copy()
    home_games["HOME"] = True  # Mark as home team

    cols_to_keep_away = ["GAME_ID", "TEAM_ID_AWAY", "GAME_DATE", "SEASON_ID"]
    if "GAME_TIME" in games_renamed.columns:
        cols_to_keep_away.append("GAME_TIME")

    away_games = games_renamed[cols_to_keep_away].copy()
    away_games.rename(columns={"TEAM_ID_AWAY": "TEAM_ID"}, inplace=True)
    away_games["HOME"] = False  # Mark as away team

    # Concatenate both home and away records
    games_expanded = pd.concat([home_games, away_games], ignore_index=True)
    team_info_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "TEAM_CITY"]
    team_info_source = df[team_info_cols].copy()
    team_info_source["TEAM_ID"] = team_info_source["TEAM_ID"].astype(str)

    # Pick the most recent metadata per TEAM_ID to avoid reviving historical names
    # (e.g., "New Jersey Nets" instead of "Brooklyn Nets") on scheduled rows.
    if "GAME_DATE" in df.columns:
        team_info_source = team_info_source.join(
            pd.to_datetime(df["GAME_DATE"], errors="coerce").rename("_GAME_DATE_SORT")
        )
        sort_cols = ["TEAM_ID", "_GAME_DATE_SORT"]
        ascending = [True, False]
        if "GAME_ID" in df.columns:
            team_info_source = team_info_source.join(df["GAME_ID"].rename("_GAME_ID_SORT"))
            sort_cols.append("_GAME_ID_SORT")
            ascending.append(False)
        team_info_source = team_info_source.sort_values(
            by=sort_cols, ascending=ascending, kind="mergesort"
        )

    team_info_from_df = (
        team_info_source.drop_duplicates(subset=["TEAM_ID"], keep="first")[team_info_cols]
        .copy()
    )

    id_to_name = {team_id: name for name, team_id in TEAM_ID_MAP.items()}
    team_info_from_df["TEAM_NAME"] = (
        team_info_from_df["TEAM_NAME"]
        .map(TEAM_NAME_STANDARDIZATION)
        .fillna(team_info_from_df["TEAM_NAME"])
        .fillna(team_info_from_df["TEAM_ID"].astype(str).map(id_to_name))
    )
    games_expanded = games_expanded.merge(team_info_from_df, on="TEAM_ID", how="left")

    # Merge with `df`, keeping columns from both dataframes
    # Preserve GAME_TIME if it exists in games_expanded
    combined_df = pd.concat([df, games_expanded], ignore_index=True, join="outer")
    columns_to_keep = list(df.columns)
    if "GAME_TIME" in combined_df.columns and "GAME_TIME" not in columns_to_keep:
        columns_to_keep.append("GAME_TIME")
    df_merged = combined_df[columns_to_keep]

    # remove duplicated rows based on TEAM_ID and GAME_DATE, keeping the last one
    df_merged = df_merged.drop_duplicates(subset=["TEAM_ID", "GAME_DATE"], keep="last").reset_index(
        drop=True
    )
    df_merged = df_merged.sort_values(by=["GAME_DATE"], ascending=False).reset_index(drop=True)

    return df_merged

def standardize_and_merge_scheduled_games_to_players_data(
    games_original, df_players_original
):
    games = games_original.copy()
    df_players = df_players_original.copy()
    games = games.rename(columns={"SEASON": "SEASON_YEAR"})

    games["SEASON_ID"] = games.apply(
        lambda x: f"{x['GAME_ID'][3]}{x['SEASON_YEAR']}", axis=1
    )
    games = games.rename(columns={"GAME_DATE_EST": "GAME_DATE"})
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    cols_to_keep = ["SEASON_ID", "SEASON_YEAR", "GAME_DATE", "GAME_ID"]

    games_home = games[cols_to_keep + ["HOME_TEAM_ID"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    games_away = games[cols_to_keep + ["VISITOR_TEAM_ID"]].rename(
        columns={"VISITOR_TEAM_ID": "TEAM_ID"}
    )

    # Combine them into a single DataFrame for "all participating teams in each game"
    games_teams = pd.concat([games_home, games_away], ignore_index=True)
    games_teams["TEAM_ID"] = games_teams["TEAM_ID"].astype(str)

    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], format="%Y-%m-%d")
    df_players = df_players.sort_values(by=["PLAYER_ID", "GAME_DATE"])

    # 3) Group by PLAYER_ID and grab the last row in each group
    df_last_game = df_players.groupby("PLAYER_ID", as_index=False).tail(1)
    df_last_game["TEAM_ID"] = df_last_game["TEAM_ID"].astype(str)

    cols_to_keep = []
    for col in df_last_game.columns:
        if col == "START_POSITION":
            break
        cols_to_keep.append(col)

    df_next_game = pd.DataFrame(columns=df_last_game.columns)
    for row in games_teams.itertuples():
        df_temp = df_last_game[df_last_game["TEAM_ID"] == row.TEAM_ID].copy()
        # set all to null except cols to keep
        for col in df_temp.columns:
            if col not in cols_to_keep:
                df_temp[col] = None

        df_temp["GAME_ID"] = row.GAME_ID
        df_temp["SEASON_ID"] = row.SEASON_ID
        df_temp["SEASON_YEAR"] = row.SEASON_YEAR
        df_temp["GAME_DATE"] = row.GAME_DATE
        df_next_game = pd.concat([df_next_game, df_temp], ignore_index=True)

    return df_next_game
