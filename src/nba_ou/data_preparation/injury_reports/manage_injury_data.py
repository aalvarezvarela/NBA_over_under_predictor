from collections import defaultdict

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players, teams
from nba_ou.config.constants import TEAM_ID_MAP as TEAM_CONVERSION_DICT


# Supporting functions
def convert_name(name):
    if name and "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name


def get_player_id(player_name, team_name=None):
    if not player_name:
        return None

    matched_players = players.find_players_by_full_name(player_name)
    if not matched_players:
        return None

    if team_name and len(matched_players) > 1:
        team_list = teams.find_teams_by_full_name(team_name)
        if not team_list:
            return None
        team_id = team_list[0]["id"]

        for player in matched_players:
            player_info = commonplayerinfo.CommonPlayerInfo(
                player_id=player["id"]
            ).get_normalized_dict()
            player_team_id = player_info["CommonPlayerInfo"][0]["TEAM_ID"]
            if player_team_id == team_id:
                return str(player["id"])

        return None

    return str(matched_players[0]["id"])


def get_out_players_by_game_and_team(df: pd.DataFrame) -> dict:
    out_df = df[
        (df["Current Status"] == "Out")
        & df["GAME_ID"].notna()
        & df["Player ID"].notna()
    ]
    result = defaultdict(lambda: defaultdict(list))
    for _, row in out_df.iterrows():
        game_id = row["GAME_ID"]
        team_id = row["Team_ID"]
        player_id = row["Player ID"]
        if player_id and team_id and game_id:
            result[game_id][team_id].append(player_id)

    return {game: dict(teams) for game, teams in result.items()}


def process_injury_data(games: pd.DataFrame, injur_df: pd.DataFrame):
    # Ensure datetime
    injur_df["Game Date"] = pd.to_datetime(injur_df["Game Date"])
    games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])

    # Create merge keys
    injur_df["merge_key"] = injur_df["Game Date"].dt.strftime("%Y%m%d") + injur_df[
        "Matchup"
    ].str.replace("@", "", regex=False)
    games["merge_key"] = games["GAMECODE"].str.replace("/", "", regex=False)

    # Merge to get GAME_ID
    injur_df = injur_df.merge(
        games[["merge_key", "GAME_ID"]], on="merge_key", how="left"
    )

    # Convert player names from "Last, First" to "First Last"
    injur_df["Player Name"] = injur_df["Player Name"].apply(convert_name)

    # Get Player ID
    injur_df["Player ID"] = injur_df.apply(
        lambda row: get_player_id(row["Player Name"], row["Team"]), axis=1
    )

    injur_df["Team_ID"] = injur_df["Team"].map(TEAM_CONVERSION_DICT)
    # Extract out players
    out_players_dict = get_out_players_by_game_and_team(injur_df)

    # Identify not-yet-submitted games
    games_not_yet_submitted = (
        injur_df.loc[injur_df["Reason"] == "NOT YET SUBMITTED", "GAME_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    return out_players_dict, games_not_yet_submitted
