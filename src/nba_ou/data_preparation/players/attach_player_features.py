import pandas as pd
from tqdm import tqdm

from nba_ou.data_preparation.past_injuries.past_injuries import (
    N_TOP_PLAYERS_INJURED,
    N_TOP_PLAYERS_NON_INJURED,
    create_player_lookup,
    get_injured_players_dict,
)
from nba_ou.data_preparation.players.players_statistics import (
    get_top_n_averages_with_names,
    precompute_cumulative_avg_stat,
)


def clear_player_statistics(df_players, df_team):
    """
    Process player statistics and prepare for training.

    This function handles:
    - Merging player data with game dates from team data
    - Converting player minutes from MM:SS format to decimal
    - Cleaning and deduplicating player data

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        df (pd.DataFrame): Processed team DataFrame with GAME_ID, GAME_DATE, SEASON_ID

    Returns:
        pd.DataFrame: Processed player DataFrame
    """
    df_players = df_players.merge(
        df_team[["GAME_ID", "GAME_DATE", "SEASON_ID"]],
        on="GAME_ID",
        how="left",
    )
    df_players = df_players.dropna(subset=["GAME_DATE"])

    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], format="%Y-%m-%d")
    df_players["MIN"] = df_players["MIN"].astype(str)

    # Extract minutes (XX) and seconds (YY) separately
    df_players[["MINUTES", "SECONDS"]] = df_players["MIN"].str.extract(
        r"^(\d+\.?\d*):?(\d*)$"
    )

    # Convert to float (handle empty second values as 0)
    df_players["MINUTES"] = df_players["MINUTES"].astype(float)
    df_players["SECONDS"] = df_players["SECONDS"].replace("", 0).astype(float)

    # Compute total playing time in minutes
    df_players["MIN"] = df_players["MINUTES"] + (df_players["SECONDS"] / 60)
    df_players["MIN"] = df_players["MIN"].round(3).fillna(0)

    # Drop temporary columns
    df_players.drop(columns=["MINUTES", "SECONDS"], inplace=True)

    df_players = df_players.drop_duplicates(keep="first")

    return df_players


def add_player_history_features(df_team, df_players, df_injuries, stat_cols=["PTS"]):
    """
    Main function to attach top player statistics and injured player stats to team data.

    - Precomputes cumulative avg of `stat_cols` in df_players
    - Builds an injured_dict from df_injuries
    - For each row in df_team, finds top-n average of `stat_cols` among non-injured
      and injured players who belong to that team on that date

    Args:
        df_team (pd.DataFrame): Team-level game data
        df_players (pd.DataFrame): Player-level boxscore data
        df_injuries (pd.DataFrame): Injury data per (GAME_ID, TEAM_ID, PLAYER_ID)
        stat_cols (list or str): Statistics columns to compute averages for

    Returns:
        pd.DataFrame: Updated df_team with extra columns for top players and injured players
    """
    # Build injuries lookup
    injured_dict = get_injured_players_dict(df_injuries)

    if isinstance(stat_cols, str):
        stat_cols = [stat_cols]

    # Collect all column names first to avoid fragmentation
    all_new_cols = []
    for stat_col in stat_cols:
        # 1) Precompute cumulative averages for the chosen stat
        df_players = precompute_cumulative_avg_stat(df_players, stat_col=stat_col)

        # 2) Dynamically name new columns based on `stat_col`
        new_cols = [
            *[
                f"TOP{i}_PLAYER_ID_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
            ],
            *[
                f"TOP{i}_PLAYER_NAME_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
            ],
            *[
                f"TOP{i}_PLAYER_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_ID_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_NAME_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
            ],
            f"AVG_INJURED_{stat_col}",
        ]
        all_new_cols.extend(new_cols)

    # Create all columns at once to avoid fragmentation
    new_cols_df = pd.DataFrame(None, index=df_team.index, columns=all_new_cols)
    df_team = pd.concat([df_team, new_cols_df], axis=1)

    # Sort df_players once before the loop for optimal performance
    df_players = df_players.copy()
    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], errors="coerce")
    df_players.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort", inplace=True)

    # Create optimized lookup function (precomputes indexes once)
    player_lookup = create_player_lookup(df_players)

    # 3) Iterate over each row in df_team (only needed columns for efficiency)
    cols_needed = ["GAME_ID", "TEAM_ID", "SEASON_ID", "GAME_DATE"]

    # Collect all updates in a list for bulk assignment
    updates_list = []

    for idx, (game_id, team_id, season_id, game_date) in enumerate(
        tqdm(
            df_team[cols_needed].itertuples(index=False, name=None),
            total=len(df_team),
            desc="Adding players data",
        )
    ):
        # Identify active players using optimized lookup
        df_active = player_lookup(season_id, team_id, game_date)
        if df_active.empty:
            updates_list.append({})
            continue

        # Who is injured for this game/team?
        game_injured_map = injured_dict.get(game_id, {})
        injured_players = set(game_injured_map.get(team_id, []))

        # Separate non-injured and injured players
        df_non_inj = df_active[~df_active["PLAYER_ID"].isin(injured_players)]
        df_inj = df_active[df_active["PLAYER_ID"].isin(injured_players)]

        row_update = {}
        for stat_col in stat_cols:
            # Top-n for each (use module constants)
            n_players_noninj = N_TOP_PLAYERS_NON_INJURED
            n_players_inj = N_TOP_PLAYERS_INJURED

            topn_non_inj = get_top_n_averages_with_names(
                df_non_inj,
                date=game_date,
                stat_col=stat_col,
                injured=False,
                n_players=n_players_noninj,
            )
            top3_inj = get_top_n_averages_with_names(
                df_inj,
                date=game_date,
                stat_col=stat_col,
                n_players=n_players_inj,
                injured=True,
            )

            # Pad to required length with None for IDs/names, 0 for stats
            while len(topn_non_inj) < n_players_noninj:
                topn_non_inj.append((None, None, 0))
            while len(top3_inj) < n_players_inj:
                top3_inj.append((None, None, 0))

            for i in range(n_players_noninj):
                row_update[f"TOP{i+1}_PLAYER_ID_{stat_col}"] = topn_non_inj[i][0]
                row_update[f"TOP{i+1}_PLAYER_NAME_{stat_col}"] = topn_non_inj[i][1]
                row_update[f"TOP{i+1}_PLAYER_{stat_col}"] = topn_non_inj[i][2]

            for i in range(n_players_inj):
                row_update[f"TOP{i+1}_INJURED_PLAYER_ID_{stat_col}"] = top3_inj[i][0]
                row_update[f"TOP{i+1}_INJURED_PLAYER_NAME_{stat_col}"] = top3_inj[i][1]
                row_update[f"TOP{i+1}_INJURED_PLAYER_{stat_col}"] = top3_inj[i][2]

            inj_values = [val for (_, _, val) in top3_inj if val != 0]
            row_update[f"AVG_INJURED_{stat_col}"] = (
                sum(inj_values) / len(inj_values) if inj_values else 0
            )

        updates_list.append(row_update)

    # Apply all updates at once using a DataFrame
    updates_df = pd.DataFrame(updates_list, index=df_team.index)
    for col in updates_df.columns:
        df_team[col] = updates_df[col]

    df_team["TOTAL_INJURED_PLAYER_PTS_BEFORE"] = (
        df_team[
            [
                "TOP1_INJURED_PLAYER_PTS",
                "TOP2_INJURED_PLAYER_PTS",
                "TOP3_INJURED_PLAYER_PTS",
            ]
        ]
        .sum(axis=1, skipna=True)
        .fillna(0)
    )

    return df_team, injured_dict
