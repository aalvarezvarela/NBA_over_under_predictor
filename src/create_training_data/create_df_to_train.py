"""
NBA Over/Under Predictor - Training Data Creation Module

This module creates training datasets for NBA over/under prediction models.
It processes historical data from the last two seasons, computing all features
and statistics needed for model training, including injury data processing.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))
from data_processing.process_data_with_injuries import (
    add_last_season_playoff_games,
    compute_differences_in_points_conceeded_anotated,
    compute_home_points_conceded_avg,
    compute_trend_slope,
    get_last_5_matchup_excluding_current,
)
from data_processing.process_referee_data import (
    add_referee_features_to_training_data,
)
from data_processing.statistics import (
    classify_season_type,
    compute_rolling_stats,
    compute_rolling_weighted_stats,
    compute_season_std,
)
from data_processing.travel_processing import compute_travel_features
from fetch_data.manage_odds_data.update_odds_utils import (
    load_odds_data,
    merge_teams_df_with_odds,
)
from postgre_DB import load_all_nba_data_from_db
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def precompute_cumulative_avg_stat(df_players, stat_col="PTS"):
    """
    Compute a cumulative average of the specified stat per (SEASON_ID, PLAYER_ID),
    considering only past games where they played (MIN > 0).

    - Excludes the current game's stat from the average.
    - If a player has not played before, their cumulative average is 0.
    - Only considers games where MIN > 0 as a valid appearance.
    - Groups by both SEASON_ID and PLAYER_ID.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        stat_col (str): Column name for the statistic to compute average for

    Returns:
        pd.DataFrame: Updated DataFrame with cumulative average columns
    """
    # 1) Sort by season, player, then ascending game date
    df_players = df_players.sort_values(
        ["SEASON_ID", "PLAYER_ID", "GAME_DATE"], ascending=True
    ).copy()

    # 2) Drop rows with NaN for the given stat
    df_players = df_players.dropna(subset=[stat_col])

    # 3) Identify valid games (where the player actually played)
    df_players["VALID_GAME"] = (df_players["MIN"] > 0).astype(int)

    # 4) Shift the stat so we only use past games
    shifted_stat_col = f"{stat_col}_PREV"
    df_players[shifted_stat_col] = df_players.groupby(["SEASON_ID", "PLAYER_ID"])[
        stat_col
    ].shift(1)
    df_players["VALID_GAME_PREV"] = (
        df_players.groupby(["SEASON_ID", "PLAYER_ID"])["VALID_GAME"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # 5) Compute cumulative sum of the stat (only from shifted "past" values)
    df_players[f"CUMSUM_{stat_col}"] = (
        df_players.groupby(["SEASON_ID", "PLAYER_ID"])[shifted_stat_col]
        .cumsum()
        .fillna(0)
    )

    # 6) Compute cumulative count of valid (past) games
    df_players["GAME_COUNT"] = (
        df_players.groupby(["SEASON_ID", "PLAYER_ID"])["VALID_GAME_PREV"]
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    # 7) Compute cumulative average
    df_players[f"{stat_col}_CUM_AVG"] = (
        df_players[f"CUMSUM_{stat_col}"] / df_players["GAME_COUNT"]
    )

    # Replace averages with 0 where GAME_COUNT is 0 (player had no previous games)
    df_players.loc[df_players["GAME_COUNT"] == 0, f"{stat_col}_CUM_AVG"] = 0

    # 8) Drop helper columns
    df_players.drop(
        columns=[shifted_stat_col, "VALID_GAME", "VALID_GAME_PREV"], inplace=True
    )

    return df_players


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

    # Drop rows with NaN points
    df_result = df_result.dropna(subset=["PTS"])

    return df_result


def get_top_n_averages_with_names(
    df, date, stat_col="PTS", injured=False, lowest=False, n_players=3, min_minutes=15
):
    """
    Returns a list of tuples (Player Name, <CUM_AVG>) for the top n (or bottom n) players
    by {stat_col}_CUM_AVG.

    Args:
        df (pd.DataFrame): DataFrame with player stats including cumulative averages
        date (datetime or str): The target date (usually the current game date)
        stat_col (str): The stat column (e.g., "PTS") for cumulative average lookup
        injured (bool): If False, consider players who played on `date`.
                       If True, consider last game prior to `date`
        lowest (bool): If False (default), return highest averages (descending).
                      If True, return lowest averages (ascending)
        n_players (int): Number of players to return
        min_minutes (int): Minimum average minutes threshold

    Returns:
        list: List of tuples (player_name, cumulative_average)
    """
    if stat_col == "DEF_RATING":
        lowest = True

    if injured:
        min_minutes = min_minutes * 0.8

    if df.empty:
        return []

    # Ensure df is sorted by date so tail(1) is truly the last prior game
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=True).copy()

    if injured:
        # For injured players: last game *before* `date`
        df_inj = df[df["GAME_DATE"] < date]
        df_last = df_inj.groupby("PLAYER_ID", as_index=False).tail(1)
    else:
        # For non-injured: look at the game exactly on `date`
        df_last = df[df["GAME_DATE"] == date]

    if df_last.empty:
        return []

    df_cum_min = (
        df[df["GAME_DATE"] < date]
        .groupby("PLAYER_ID", as_index=False)["MIN"]
        .mean()
        .rename(columns={"MIN": "MIN_CUM_AVG"})
    )

    # Merge the cumulative average minutes into the selected game rows
    df_last = df_last.merge(df_cum_min, on="PLAYER_ID", how="left")

    cum_col = f"{stat_col}_CUM_AVG"

    # Create extra variable to check if player meets the minimum threshold
    df_last["MEETS_MIN_THRESHOLD"] = (df_last["MIN_CUM_AVG"] >= min_minutes).astype(int)

    # Sort by the cumulative average column
    df_sorted = df_last.sort_values(
        by=["MEETS_MIN_THRESHOLD", cum_col], ascending=[False, lowest]
    )

    # Extract the top n (or bottom n) players
    chosen = df_sorted.head(n_players)

    top_or_bottom_n = list(zip(chosen["PLAYER_NAME"], chosen[cum_col]))

    return top_or_bottom_n


def attach_top3_stats(df_team, df_players, df_injuries, stat_cols=["PTS"]):
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

    for stat_col in stat_cols:
        # 1) Precompute cumulative averages for the chosen stat
        df_players = precompute_cumulative_avg_stat(df_players, stat_col=stat_col)

        # 2) Dynamically name new columns based on `stat_col`
        new_cols = [
            *[f"TOP{i}_PLAYER_NAME_{stat_col}" for i in range(1, 7)],
            *[f"TOP{i}_PLAYER_{stat_col}" for i in range(1, 7)],
            *[f"TOP{i}_INJURED_PLAYER_NAME_{stat_col}" for i in range(1, 4)],
            *[f"TOP{i}_INJURED_PLAYER_{stat_col}" for i in range(1, 4)],
            f"AVG_INJURED_{stat_col}",
        ]

        for col in new_cols:
            df_team[col] = None

    # 3) Iterate over each row in df_team
    for idx, row in tqdm(
        df_team.iterrows(), total=df_team.shape[0], desc="Adding players data"
    ):
        game_id = row["GAME_ID"]
        team_id = row["TEAM_ID"]
        season_id = row["SEASON_ID"]
        game_date = row["GAME_DATE"]

        # Identify active players
        df_active = _get_players_for_team_in_season(
            df_players=df_players,
            season_id=season_id,
            team_id=team_id,
            date_to_filter=game_date,
        )
        if df_active.empty:
            continue

        # Who is injured for this game/team?
        game_injured_map = injured_dict.get(game_id, {})
        injured_players = set(game_injured_map.get(team_id, []))

        # Separate non-injured and injured players
        df_non_inj = df_active[~df_active["PLAYER_ID"].isin(injured_players)]
        df_inj = df_active[df_active["PLAYER_ID"].isin(injured_players)]

        for stat_col in stat_cols:
            # Top-n for each
            n_players_noninj = 6
            n_players_inj = 3

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

            # Pad to required length with None for names, 0 for stats
            while len(topn_non_inj) < n_players_noninj:
                topn_non_inj.append((None, 0))
            while len(top3_inj) < n_players_inj:
                top3_inj.append((None, 0))

            row_update = {}

            for i in range(n_players_noninj):
                row_update[f"TOP{i+1}_PLAYER_NAME_{stat_col}"] = topn_non_inj[i][0]
                row_update[f"TOP{i+1}_PLAYER_{stat_col}"] = topn_non_inj[i][1]

            for i in range(n_players_inj):
                row_update[f"TOP{i+1}_INJURED_PLAYER_NAME_{stat_col}"] = top3_inj[i][0]
                row_update[f"TOP{i+1}_INJURED_PLAYER_{stat_col}"] = top3_inj[i][1]

            inj_values = [val for (_, val) in top3_inj if val != 0]
            row_update[f"AVG_INJURED_{stat_col}"] = (
                sum(inj_values) / len(inj_values) if inj_values else 0
            )

            # Single assignment for all columns in row_update
            df_team.loc[idx, row_update.keys()] = row_update.values()

    return df_team


def process_team_and_player_statistics_for_training(df, df_players, df_odds):
    """
    Process and compute team statistics, rolling averages, and player data for training.

    This function handles:
    - Data cleaning and overtime adjustments
    - Merging game and odds data
    - Computing team records, win/loss statistics
    - Calculating rolling statistics and trends
    - Processing player minutes and game data

    Args:
        df (pd.DataFrame): Team game statistics DataFrame
        df_players (pd.DataFrame): Player statistics DataFrame
        df_odds (pd.DataFrame): Betting odds DataFrame

    Returns:
        tuple: (df, df_players) - Processed team and player DataFrames
    """
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    df.dropna(subset=["PTS"], inplace=True)

    # Convert to string TEAM_ID
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)

    # Handle overtime adjustments
    df["IS_OVERTIME"] = df["MIN"].apply(lambda x: 1 if x >= 259 else 0)
    mask_overtime = df["MIN"] >= 260

    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols_to_adjust = [
        col
        for col in numeric_cols
        if col
        not in ["MIN", "PACE_PER40", "SEASON_ID", "TEAM_ID", "GAME_ID", "IS_OVERTIME"]
    ]
    int_cols = df[cols_to_adjust].select_dtypes(include=["int64"]).columns.tolist()

    df[int_cols] = df[int_cols].astype(float)

    df.loc[mask_overtime, cols_to_adjust] = (
        df.loc[mask_overtime, cols_to_adjust]
        .astype(float)
        .apply(lambda x: x * (240 / df.loc[mask_overtime, "MIN"]), axis=0)
    )
    for col in cols_to_adjust:
        if df[col].dtype == "float64" and col in int_cols:
            df[col] = df[col].round().astype(int)

    df[int_cols] = df[int_cols].round().astype(int)
    print("Overtime adjustments completed.")

    # Remove duplicates and clean dataset
    df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="first", inplace=True)
    df.dropna(subset=["PTS"], inplace=True)
    df = df[df["MIN"] != 0]
    df = df[df["PTS"] > 10]

    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    df = merge_teams_df_with_odds(df_odds=df_odds, df_team=df)
    df["TOTAL_POINTS"] = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["DIFF_FROM_LINE"] = df["TOTAL_POINTS"] - df["TOTAL_OVER_UNDER_LINE"]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")

    valid_games = df["GAME_ID"].value_counts()
    valid_games = valid_games[valid_games == 2].index

    df = df[df["GAME_ID"].isin(valid_games)]

    df.loc[:, "SEASON_TYPE"] = df["GAME_ID"].apply(classify_season_type)
    df.loc[:, "SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)

    df = add_last_season_playoff_games(df)

    df.sort_values(by="GAME_DATE", ascending=True, inplace=True)

    group_cols = ["SEASON_TYPE", "SEASON_ID", "TEAM_ID"]

    df["GAME_NUMBER"] = df.groupby(group_cols).cumcount() + 1

    def compute_wins_losses(df, team_id, season_id, date):
        filtered_df = df[
            (df["TEAM_ID"] == team_id)
            & (df["GAME_DATE"] < date)
            & (df["SEASON_ID"] == season_id)
        ]
        wins = (filtered_df["WL"] == "W").sum()
        losses = (filtered_df["WL"] == "L").sum()
        return wins, losses

    tqdm.pandas()
    df["WINS_BEFORE_THIS_GAME"], df["LOSSES_BEFORE_THIS_GAME"] = zip(
        *df.progress_apply(
            lambda x: compute_wins_losses(
                df, x["TEAM_ID"], x["SEASON_ID"], x["GAME_DATE"]
            ),
            axis=1,
        )
    )

    df["TEAM_RECORD_BEFORE_GAME"] = df["WINS_BEFORE_THIS_GAME"] / (
        df["WINS_BEFORE_THIS_GAME"] + df["LOSSES_BEFORE_THIS_GAME"]
    )
    df["TEAM_RECORD_BEFORE_GAME"].fillna(0, inplace=True)

    # Compute rest days between matches
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["REST_DAYS_BEFORE_MATCH"] = (
        df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(0).astype(int)
    )
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    # Compute rolling statistics
    cols_to_average = [
        "PTS",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "EFG_PCT",
        "PACE_PER40",
        "FG3A",
        "FG3M",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3_PCT",
        "FTA",
        "FTM",
        "EFG_PCT",
        "TS_PCT",
        "POSS",
        "PIE",
    ]

    cols_to_average_odds = [
        "TOTAL_OVER_UNDER_LINE",
        "DIFF_FROM_LINE",
        "TOTAL_POINTS",
        "MONEYLINE",
        "SPREAD",
    ]

    for col in tqdm(
        cols_to_average + cols_to_average_odds, desc="Computing rolling stats"
    ):
        df = compute_rolling_stats(df, col, window=5, season_avg=True)
        if col == "PTS" or col == "TOTAL_POINTS" or col == "TOTAL_OVER_UNDER_LINE":
            df = compute_rolling_weighted_stats(df, col)

    df = compute_season_std(df, param="PTS")
    df = compute_season_std(df, param="TOTAL_POINTS")
    df = compute_season_std(df, param="TOTAL_OVER_UNDER_LINE")
    df = compute_season_std(df, param="DIFF_FROM_LINE")

    df_players = df_players.merge(
        df[["GAME_ID", "GAME_DATE", "SEASON_ID"]],
        on="GAME_ID",
        how="left",
    )
    df_players = df_players.dropna(subset="GAME_DATE")

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

    df_players = df_players.drop_duplicates()
    df = df.drop_duplicates()

    return df, df_players


def merge_home_away_and_prepare_training_features(df):
    """
    Merge home and away team data and prepare final training features.

    Args:
        df (pd.DataFrame): Team statistics DataFrame with injury data attached

    Returns:
        pd.DataFrame: Final training-ready DataFrame
    """
    # Rename columns that start with 'TOP' by adding '_BEFORE' at the end
    df.rename(
        columns=lambda x: f"{x}_BEFORE" if x.startswith("TOP") else x, inplace=True
    )

    # Add '_BEFORE' suffix to specified AVG_INJURED_* columns
    avg_injured_cols = [
        "AVG_INJURED_PTS",
        "AVG_INJURED_PACE_PER40",
        "AVG_INJURED_DEF_RATING",
        "AVG_INJURED_OFF_RATING",
        "AVG_INJURED_TS_PCT",
    ]

    df.rename(columns={col: f"{col}_BEFORE" for col in avg_injured_cols}, inplace=True)

    df["STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE"] = (
        df["TOP1_PLAYER_OFF_RATING_BEFORE"] / df["OFF_RATING_SEASON_BEFORE_AVG"]
    )
    df["STAR_PTS_PERCENTAGE_BEFORE"] = (
        df["TOP1_PLAYER_PTS_BEFORE"] / df["PTS_SEASON_BEFORE_AVG"]
    )

    # Fill NaNs in injured player columns with zeros
    injured_player_cols = [col for col in df.columns if "INJURED_PLAYER" in col]
    df[injured_player_cols] = df[injured_player_cols].fillna(0)

    # Merge home and away stats
    static_columns = [
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "SEASON_TYPE",
        "SEASON_YEAR",
        "IS_OVERTIME",
    ]

    df_home = df[df["HOME"]].copy().drop(columns="HOME")
    df_away = df[~df["HOME"]].copy().drop(columns="HOME")

    df_merged = pd.merge(
        df_home,
        df_away,
        on=static_columns,
        how="inner",
        suffixes=("_TEAM_HOME", "_TEAM_AWAY"),
    )
    df_merged["TOTAL_POINTS"] = df_merged.PTS_TEAM_HOME + df_merged.PTS_TEAM_AWAY

    df_merged = df_merged[
        (df_merged["SEASON_TYPE"] != "In-Season Final Game")
        & (df_merged["SEASON_TYPE"] != "All Star")
        & (df_merged["SEASON_TYPE"] != "In-Season Tournament")
    ]

    df_merged["IS_PLAYOFF_GAME"] = (
        df_merged["SEASON_ID"].astype(str).str.startswith("4").astype(int)
    )
    df_merged["TOTAL_OVER_UNDER_LINE"] = df_merged["TOTAL_OVER_UNDER_LINE_TEAM_HOME"]
    df_merged["SPREAD"] = df_merged["SPREAD_TEAM_HOME"]

    # Apply function
    df_merged = compute_home_points_conceded_avg(df_merged)
    df_merged["DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"] = (
        df_merged["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
        - df_merged["AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME"]
    )
    df_merged["DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"] = (
        df_merged["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        - df_merged["AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME"]
    )

    df_merged = compute_differences_in_points_conceeded_anotated(df_merged)
    df_merged["TEAMS_DIFFERENCE_OVER_UNDER_LINE_BEFORE"] = (
        df_merged["TOTAL_OVER_UNDER_LINE_SEASON_BEFORE_AVG_TEAM_HOME"]
        - df_merged["TOTAL_OVER_UNDER_LINE_SEASON_BEFORE_AVG_TEAM_AWAY"]
    )

    tqdm.pandas()
    # Apply row-by-row, returning a Series of dictionaries
    results_series = df_merged.progress_apply(
        lambda row: get_last_5_matchup_excluding_current(row, df_merged), axis=1
    )

    # Convert that Series of dicts into a DataFrame
    results_df = pd.DataFrame(results_series.tolist(), index=df_merged.index)

    # Finally, concatenate the new columns onto df_merged
    df_merged = pd.concat([df_merged, results_df], axis=1)
    df_merged.sort_values(["TEAM_ID_TEAM_HOME", "GAME_DATE"], ascending=True)

    # Compute trends using linear regression
    df_merged = compute_trend_slope(df_merged, parameter="PTS", window=5)
    df_merged = compute_trend_slope(df_merged, parameter="TS_PCT", window=5)
    df_merged = compute_trend_slope(
        df_merged, parameter="TOTAL_OVER_UNDER_LINE", window=5
    )
    df_merged = df_merged.sort_values(by="GAME_DATE", ascending=False)

    columns = [
        "TEAM_ID",
        "TEAM_CITY",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "MATCHUP",
        "GAME_NUMBER",
    ]

    # Generate new list with _HOME and _AWAY appended
    columns_info_before = [f"{col}_TEAM_HOME" for col in columns] + [
        f"{col}_TEAM_AWAY" for col in columns
    ]

    columns_info_before.extend(
        [
            "SEASON_ID",
            "IS_OVERTIME",
            "GAME_ID",
            "GAME_DATE",
            "SEASON_TYPE",
            "IS_PLAYOFF_GAME",
            "PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY",
            "PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME",
            "SEASON_YEAR",
        ]
    )
    # Insert columns that have BEFORE in the name
    columns_info_before.extend([col for col in df_merged.columns if "BEFORE" in col])

    odds_columns = [
        "TOTAL_OVER_UNDER_LINE",
        "SPREAD",
        "MONEYLINE_TEAM_HOME",
        "MONEYLINE_TEAM_AWAY",
    ]

    columns_info_before.extend(odds_columns)

    df_training = df_merged[columns_info_before + ["TOTAL_POINTS"]].copy()
    df_training = df_training[df_training["SEASON_TYPE"] != "Preseason"]

    df_training["TOTAL_PTS_SEASON_BEFORE_AVG"] = (
        df_training["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        + df_training["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    )

    df_training["TOTAL_PTS_LAST_GAMES_AVG"] = (
        df_training["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        + df_training["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )

    df_training["BACK_TO_BACK"] = (
        df_training["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"] == 1
    ) & (df_training["REST_DAYS_BEFORE_MATCH_TEAM_HOME"] == 1)

    df_training["DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE_MATCH"] = (
        df_training["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        - df_training["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )
    df_training["DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE_MATCH"] = (
        df_training["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
        - df_training["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
    )

    df_training.loc[df_training["MATCHUP_TEAM_HOME"] == 0, "MATCHUP_TEAM_HOME"] = (
        df_training["TEAM_ABBREVIATION_TEAM_HOME"]
        + " vs. "
        + df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
    )

    df_training.loc[df_training["MATCHUP_TEAM_AWAY"] == 0, "MATCHUP_TEAM_AWAY"] = (
        df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
        + " @ "
        + df_training["TEAM_ABBREVIATION_TEAM_HOME"]
    )

    df_training.drop(columns=["IS_OVERTIME"], inplace=True)

    # Move TOTAL_OVER_UNDER_LINE to first column
    first_col = "TOTAL_OVER_UNDER_LINE"
    df_training = df_training[
        [first_col] + [col for col in df_training.columns if col != first_col]
    ]

    return df_training


def get_all_seasons_from_2006(date_to_train_until):
    """
    Get all NBA seasons from 2006-07 until the season containing date_to_train_until.

    Args:
        date_to_train_until (datetime): The target date

    Returns:
        list: List of season strings in format "YYYY-YY" (e.g., ["2006-07", "2007-08", ...])
    """
    if isinstance(date_to_train_until, str):
        date_to_train_until = pd.to_datetime(date_to_train_until)

    # Determine the season year for the target date
    # NBA season runs from October (month 10) to June
    # If date is Jan-Jun, it's part of season that started previous year
    # If date is Jul-Dec, it's part of season that will start this year (or just ended)
    target_year = date_to_train_until.year
    target_month = date_to_train_until.month

    if target_month <= 6:
        # Jan-Jun: season started previous year
        end_season_year = target_year - 1
    else:
        # Jul-Dec: season starts this year
        end_season_year = target_year

    # Generate all seasons from 2006 to end_season_year
    seasons = []
    for year in range(2006, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons


def get_seasons_between_dates(date_from, date_to):
    """
    Get all NBA seasons between two dates (inclusive).

    Args:
        date_from (datetime or str): The start date
        date_to (datetime or str): The end date

    Returns:
        list: List of season strings in format "YYYY-YY" (e.g., ["2006-07", "2007-08", ...])
    """
    if isinstance(date_from, str):
        date_from = pd.to_datetime(date_from)
    if isinstance(date_to, str):
        date_to = pd.to_datetime(date_to)

    # Helper function to determine season year from a date
    def get_season_year(date):
        year = date.year
        month = date.month
        # If date is Jan-Jun, season started previous year
        # If date is Jul-Dec, season starts this year
        return year - 1 if month <= 6 else year

    start_season_year = get_season_year(date_from)
    end_season_year = get_season_year(date_to)

    # Generate all seasons between start and end
    seasons = []
    for year in range(start_season_year, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons


def load_injury_data_from_db(seasons):
    """
    Load injury data from database for the specified seasons.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])

    Returns:
        pd.DataFrame: Combined injury data for all seasons
    """
    from postgre_DB.db_config import connect_nba_db, get_schema_name_injuries
    from psycopg import sql

    schema = get_schema_name_injuries()
    table = "nba_injuries"  # table name in injuries schema

    conn = None
    try:
        conn = connect_nba_db()

        # Convert season format from "2023-24" to 2023 (year only)
        season_years = [int(s.split("-")[0]) for s in seasons]

        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            WHERE season_year = ANY(%s)
        """).format(sql.Identifier(schema), sql.Identifier(table))

        query = query_obj.as_string(conn)
        df_injuries = pd.read_sql_query(query, conn, params=(season_years,))

        # Convert column names to uppercase to match expected format
        df_injuries.columns = df_injuries.columns.str.upper()

        # Ensure TEAM_ID and PLAYER_ID are strings
        if "TEAM_ID" in df_injuries.columns:
            df_injuries["TEAM_ID"] = df_injuries["TEAM_ID"].astype(str)
        if "PLAYER_ID" in df_injuries.columns:
            df_injuries["PLAYER_ID"] = df_injuries["PLAYER_ID"].astype(str)

        # Remove duplicates
        df_injuries = df_injuries.drop_duplicates()

        print(f"Loaded {len(df_injuries)} injury records from database")
        return df_injuries

    except Exception as e:
        print(f"Error loading injury data from database: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    finally:
        if conn is not None:
            conn.close()


def create_df_to_train(
    date_to_train_until: str | datetime,
    date_from: str | datetime = None,
):
    """
    Create training dataset for NBA over/under prediction models.

    This function:
    - Loads data from database for seasons from date_from (or 2006-07) to date_to_train_until
    - Processes injuries from database (not from live reports)
    - Computes all team and player statistics
    - Calculates rolling averages and trends
    - Merges home/away data and prepares final training features

    Args:
        date_to_train_until (str | datetime): Latest date to include in training data (YYYY-MM-DD)
        df_odds (pd.DataFrame): Betting odds data
        date_from (str | datetime, optional): Starting date for training data. If None, starts from 2006-07

    Returns:
        pd.DataFrame: Complete training dataset with all features
    """
    df_odds = load_odds_data()

    if isinstance(date_to_train_until, str):
        date_to_train_until = pd.to_datetime(date_to_train_until, format="%Y-%m-%d")
    else:
        date_to_train_until = pd.to_datetime(date_to_train_until)

    # Determine which seasons to load
    if date_from is not None:
        if isinstance(date_from, str):
            date_from = pd.to_datetime(date_from, format="%Y-%m-%d")
        else:
            date_from = pd.to_datetime(date_from)
        seasons = get_seasons_between_dates(date_from, date_to_train_until)
    else:
        seasons = get_all_seasons_from_2006(date_to_train_until)

    print(f"Loading data for seasons: {seasons}")

    # Load game and player data from database
    df, df_players = load_all_nba_data_from_db(seasons=seasons)

    # Ensure GAME_DATE column is pandas Timestamp for df (df_players doesn't have it yet)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Filter df by date range
    if date_from is not None:
        df = df[
            (df["GAME_DATE"] >= date_from) & (df["GAME_DATE"] <= date_to_train_until)
        ]
    else:
        df = df[df["GAME_DATE"] <= date_to_train_until]

    print(f"Loaded {len(df)} team game records and {len(df_players)} player records")

    # Process team and player statistics (this adds GAME_DATE to df_players)
    df, df_players = process_team_and_player_statistics_for_training(
        df, df_players, df_odds
    )

    # Now filter df_players by date range (after GAME_DATE has been added via merge)
    if date_from is not None:
        df_players = df_players[
            (df_players["GAME_DATE"] >= date_from)
            & (df_players["GAME_DATE"] <= date_to_train_until)
        ]
    else:
        df_players = df_players[df_players["GAME_DATE"] <= date_to_train_until]

    # Load injury data from database
    df_injuries = load_injury_data_from_db(seasons)

    print(f"Loaded {len(df_injuries)} injury records")

    # Define statistics to compute for top players
    stats = ["PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT"]

    # Attach top player statistics including injury data
    df = attach_top3_stats(df, df_players, df_injuries, stats)

    # Sum injured players' points into a new column
    df["TOTAL_INJURED_PLAYER_PTS_BEFORE"] = (
        df[
            [
                "TOP1_INJURED_PLAYER_PTS",
                "TOP2_INJURED_PLAYER_PTS",
                "TOP3_INJURED_PLAYER_PTS",
            ]
        ]
        .sum(axis=1, skipna=True)
        .fillna(0)
    )

    # Merge home and away data and prepare final features
    df_training = merge_home_away_and_prepare_training_features(df)

    print()
    print("--" * 20)
    print(f"Training data created up to {date_to_train_until}")
    print(f"Number of games: {df_training.shape[0]}")
    print(f"Number of features: {df_training.shape[1]}")
    print("--" * 20)
    print()

    return df_training


if __name__ == "__main__":
    # Example usage

    # Create training data up to a specific date
    date_to_train = "2025-01-10"
    date_from = "2006-11-01"  # Optional: specify start date

    df_train = create_df_to_train(
        date_to_train_until=date_to_train, date_from=date_from
    )

    # Load referee data from database
    if date_from is not None:
        seasons = get_seasons_between_dates(date_from, date_to_train)
    else:
        seasons = get_all_seasons_from_2006(date_to_train)

    df_train = add_referee_features_to_training_data(seasons, df_train)

    # Compute travel features (distance traveled in last 7 and 14 days)
    df_train = compute_travel_features(df_train)

    # Save to file with seasons in filename
    date_from_dt = (
        pd.to_datetime(date_from) if date_from else pd.to_datetime("2006-01-01")
    )
    date_to_train_dt = pd.to_datetime(date_to_train)
    output_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data"
    output_name = f"{output_path}/training_data_{date_from_dt.strftime('%Y%m%d')}_to_{date_to_train_dt.strftime('%Y%m%d')}.csv"
    df_train.to_csv(output_name, index=False)
    print(f"Training data saved to {output_name}")