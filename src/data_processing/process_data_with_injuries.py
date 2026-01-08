"""
NBA Over/Under Predictor - Main Data Processing Module

This module contains the main data processing pipeline for NBA game predictions,
including injury data integration, statistical computations, and feature engineering.
"""

import os
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from fetch_data.manage_odds_data.update_odds_utils import merge_teams_df_with_odds
from nba_api.stats.endpoints import ScoreboardV2
from postgre_DB import load_all_nba_data_from_db
from scipy.stats import linregress
from tqdm import tqdm

from .injury_processing import process_injury_data, retrieve_injury_report_as_df
from .statistics import (
    attach_top3_stats,
    classify_season_type,
    compute_rolling_stats,
    compute_rolling_weighted_stats,
    compute_season_std,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_schedule_games(date):
    """
    Fetches NBA scheduled games for a given date.

    Parameters:
    date (str): The date in 'YYYY-MM-DD' format.

    Returns:
    DataFrame: A DataFrame containing scheduled games for the given date.
    """
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")

    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

    # Fetch games
    scoreboard_v2 = ScoreboardV2(game_date=date)

    games = scoreboard_v2.get_data_frames()[0]

    return games



def get_last_two_nba_seasons(d: str | datetime) -> list[str]:
    """
    Return the current NBA season (for the given date) and the immediately previous season.

    NBA season boundary:
      - Months Jan–Aug belong to the season that started the previous calendar year.
      - Months Sep–Dec belong to the season that starts the current calendar year.

    Examples:
      - 2025-01-10 -> ["2024-25", "2023-24"]
      - 2025-10-10 -> ["2025-26", "2024-25"]
    """
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    elif isinstance(d, datetime):
        d = d.date()

    start_year = d.year - 1 if d.month < 9 else d.year
    season = f"{start_year}-{str(start_year + 1)[-2:]}"
    previous_season = f"{start_year - 1}-{str(start_year)[-2:]}"

    return [season, previous_season]

def load_all_nba_data(data_path, seasons=None):
    """
    Loads NBA game and player data for the specified seasons.

    Parameters:
        data_path (str): Path to the directory containing NBA CSV files.
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"]).

    Returns:
        tuple: (df_games, df_players) where:
            - df_games is the combined DataFrame of all loaded games.
            - df_players is the combined DataFrame of all loaded players.
    """
    game_dfs = []
    player_dfs = []

    file_list = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    if not file_list:
        print("No CSV files found in the directory.")
        return None, None
    if seasons is None:
        # Detect seasons by parsing game file names
        seasons = []
        for f in file_list:
            if f.startswith("nba_games_") and f.endswith(".csv"):
                tag = f[len("nba_games_") : -len(".csv")]
                season = tag.replace("_", "-")
                seasons.append(season)
        seasons = sorted(set(seasons))  # Remove duplicates, sort

    for season in seasons:
        season_tag = season.replace("-", "_")
        game_filename = f"nba_games_{season_tag}.csv"
        player_filename = f"nba_players_{season_tag}.csv"

        # Load games
        if game_filename in file_list:
            try:
                team_path = os.path.join(data_path, game_filename)
                df_temp = pd.read_csv(team_path, nrows=0)
                dtype_dict_teams = {
                    col: str for col in df_temp.columns if "ID" in col.upper()
                }
                df_game = pd.read_csv(team_path, dtype=dtype_dict_teams)
                game_dfs.append(df_game)
                print(f"Loaded game data: {game_filename} ({df_game.shape[0]} rows)")

            except Exception as e:
                print(f"Error loading game data {game_filename}: {e}")
        else:
            print(f"Game file not found for season {season}")

        # Load players
        if player_filename in file_list:
            player_path = os.path.join(data_path, player_filename)
            try:
                # Dynamically detect columns with "ID" and load as str
                df_temp = pd.read_csv(player_path, nrows=0)
                dtype_dict_players = {
                    col: str for col in df_temp.columns if "ID" in col.upper()
                }

                df_player = pd.read_csv(player_path, dtype=dtype_dict_players)
                player_dfs.append(df_player)
                print(
                    f"Loaded player data: {player_filename} ({df_player.shape[0]} rows)"
                )

            except Exception as e:
                print(f"Error loading player data {player_filename}: {e}")
        else:
            print(f"Player file not found for season {season}")

    df_games = pd.concat(game_dfs, ignore_index=True) if game_dfs else None
    df_players = pd.concat(player_dfs, ignore_index=True) if player_dfs else None

    return df_games, df_players


def standardize_and_merge_nba_data(df, games):
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
    games_renamed = games.rename(
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
    team_info_from_df = df[
        [
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "TEAM_NAME",
            "TEAM_CITY",
        ]
    ].drop_duplicates()
    games_expanded = games_expanded.merge(team_info_from_df, on="TEAM_ID", how="left")

    # Merge with `df`, keeping columns from both dataframes
    # Preserve GAME_TIME if it exists in games_expanded
    combined_df = pd.concat([df, games_expanded], ignore_index=True, join="outer")
    columns_to_keep = list(df.columns)
    if "GAME_TIME" in combined_df.columns and "GAME_TIME" not in columns_to_keep:
        columns_to_keep.append("GAME_TIME")
    df = combined_df[columns_to_keep]

    # remove duplicated rows based on TEAM_ID and GAME_DATE, keeping the last one
    df = df.drop_duplicates(subset=["TEAM_ID", "GAME_DATE"], keep="last").reset_index(
        drop=True
    )

    return df


def get_last_5_matchup_excluding_current(row, df):
    """
    For the given row (which represents one game),
    find the 5 most recent PRIOR matchups between (TEAM_ID_HOME vs TEAM_ID_AWAY).
    Returns a dict with 5 keys: the TOTAL_POINTS in each of those matchups,
    sorted descending by GAME_DATE, excluding the current row's game_date.

    'df' should have columns:
        - TEAM_ID_HOME
        - TEAM_ID_AWAY
        - GAME_DATE
        - PTS_HOME
        - PTS_AWAYdf_merged
    """
    home_team = row["TEAM_ID_TEAM_HOME"]
    away_team = row["TEAM_ID_TEAM_AWAY"]
    current_date = row["GAME_DATE"]

    df_matchups = df[
        (df["TEAM_ID_TEAM_HOME"] == home_team)
        & (df["TEAM_ID_TEAM_AWAY"] == away_team)
        & (df["GAME_DATE"] < current_date)
    ].copy()

    df_matchups.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    df_matchups = df_matchups.head(5)

    totals_list = df_matchups["TOTAL_POINTS"].tolist()

    if len(totals_list) == 0:
        totals_list = [np.nan] * 5
    elif len(totals_list) < 5:
        while len(totals_list) < 5:
            # append the mean of totals_list
            totals_list.append(sum(totals_list) / len(totals_list))

    mean = sum(totals_list) / 5
    return {
        "LAST_1_GAMES_TOTAL_POINTS_BEFORE": totals_list[0],
        "LAST_2_GAMES_TOTAL_POINTS_BEFORE": totals_list[1],
        "LAST_3_GAMES_TOTAL_POINTS_BEFORE": totals_list[2],
        "LAST_4_GAMES_TOTAL_POINTS_BEFORE": totals_list[3],
        "LAST_5_GAMES_TOTAL_POINTS_BEFORE": totals_list[4],
        "LAST_5_GAMES_TOTAL_POINTS_BEFORE_MEAN": mean,
    }


# Compute number of playoff games in last season
def add_last_season_playoff_games(df):
    """
    Adds a column `PLAYOFF_GAMES_LAST_SEASON` to indicate the number of playoff games
    each team played in the previous season.

    Handles SEASON_ID variations (e.g., 42024, 22024) by extracting the last 4 digits (year).

    Args:
        df (pd.DataFrame): Must contain columns `SEASON_ID`, `TEAM_ID`.

    Returns:
        pd.DataFrame: The modified DataFrame with the added column.
    """
    last_season = df["SEASON_YEAR"].max() - 1
    df_playoffs_last_season = df[
        (df["SEASON_YEAR"] == last_season)
        & (df["SEASON_ID"].astype(str).str.startswith("4"))
    ]
    playoff_counts = (
        df_playoffs_last_season.groupby("TEAM_ID")["GAME_ID"].nunique().reset_index()
    )
    playoff_counts.rename(
        columns={"GAME_ID": "PLAYOFF_GAMES_LAST_SEASON"}, inplace=True
    )
    df = df.merge(playoff_counts, on="TEAM_ID", how="left").fillna(0)
    return df


def compute_home_points_conceded_avg(df):
    """
    Computes the average points conceded by the home team when playing at home,
    in all games prior to the current one, within the same season.

    Args:
        df (pd.DataFrame): Must contain columns:
            - TEAM_ID_TEAM_HOME, SEASON_ID, GAME_DATE, PTS_TEAM_AWAY

    Returns:
        pd.DataFrame: The modified DataFrame with a new column.
    """

    # Sort DataFrame so previous games come first
    df = df.sort_values(
        ["TEAM_ID_TEAM_HOME", "SEASON_YEAR", "GAME_DATE"], ascending=True
    )

    # Define the new column name
    col_name = "AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME"

    # Compute rolling average of points conceded at home (excluding current game)
    df[col_name] = df.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
        "PTS_TEAM_AWAY"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Sort again for away calculations
    df = df.sort_values(
        ["SEASON_YEAR", "TEAM_ID_TEAM_AWAY", "GAME_DATE"], ascending=True
    )

    # Compute rolling average of points conceded away by the away team
    df["AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME"] = df.groupby(
        ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"]
    )["PTS_TEAM_HOME"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.sort_values("GAME_DATE", ascending=False)
    return df


def compute_differences_in_points_conceeded_anotated(df):
    """
    Computes the average points conceded by the home team when playing at home,
    in all games prior to the current one, within the same season.

    Args:
        df (pd.DataFrame): Must contain columns:
            - TEAM_ID_TEAM_HOME, SEASON_ID, GAME_DATE, DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME, DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME

    Returns:
        pd.DataFrame: The modified DataFrame with a new column.
    """

    # Sort DataFrame so previous games come first
    df = df.sort_values(["TEAM_ID_TEAM_HOME", "GAME_DATE"], ascending=True)

    # Define the new column name
    col_name = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_HOME"

    # Compute rolling average of points conceded at home (excluding current game)
    df[col_name] = df.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
        "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df[col_name] = df[col_name].fillna(0)
    # Sort again for away calculations
    df = df.sort_values(["TEAM_ID_TEAM_AWAY", "GAME_DATE"], ascending=True)

    col_name = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_AWAY"
    # Compute rolling average of points conceded away by the away team
    df[col_name] = df.groupby(["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"])[
        "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.sort_values("GAME_DATE", ascending=False)
    df[col_name] = df[col_name].fillna(0)

    return df


def compute_trend_slope(df, parameter="PTS", window=10):
    """
    Computes the slope of a linear regression line over the last `window` games
    to determine whether a team's performance is increasing, decreasing, or stable.

    Args:
        df (pd.DataFrame): Must contain columns "TEAM_ID_TEAM_HOME", "GAME_DATE", and `param`.
        param (str): The statistic to analyze (e.g., "PTS").
        window (int): Number of last games to consider.

    Returns:
        pd.DataFrame: A modified DataFrame with a new column:
            - f"{param}_TREND_SLOPE_LAST_{window}_GAMES"
    """

    def calculate_slope(series):
        """Applies linear regression to compute the trend slope."""
        if len(series) < 2:
            return 0  # Not enough data for a trend, so we assing 0

        X = np.arange(1, len(series) + 1)  # Time index [1, 2, ..., N]
        Y = series  # Directly use the series

        slope, _, _, _, _ = linregress(X, Y)
        return slope

    for field in ["TEAM_HOME", "TEAM_AWAY"]:
        param = f"{parameter}_{field}"

        trend_col = f"{param}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"

        # Sort games by date in ascending order
        df = df.sort_values([f"TEAM_ID_{field}", "GAME_DATE"], ascending=True)

        # Apply the function per team, shifting by 1 to exclude the current game
        trend_series = (
            df.groupby(f"TEAM_ID_{field}")[param]
            .apply(
                lambda s: s.shift(1)
                .rolling(window, min_periods=2)
                .apply(calculate_slope, raw=True)
            )
            .reset_index(level=0, drop=True)
        )  # Reset index to align with df

        # Assign the result back to the DataFrame
        df[trend_col] = trend_series
    df[f"{parameter}_COMBINED_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"] = (
        df[f"{parameter}_TEAM_HOME_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"]
        + df[f"{parameter}_TEAM_AWAY_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"]
    )
    return df


def create_df_players_new_game(games_original, df_players_original):
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


def create_df_to_predict(
    data_path: str,
    date_to_predict: str | datetime,
    nba_injury_reports_url: str,
    df_odds,
    reports_path: str = None,
    filter_for_date_to_predict: bool = True,
):
    """
    Process all NBA data for the last two seasons and return a DataFrame with the processed data.
    Args:
        data_path (str): Path to the directory containing NBA season CSV files.
        date_to_predict (str): The date in 'YYYY-MM-DD' format.
        filter_for_date_to_predict (bool): If True, filter the output DataFrame to only include games for the specified date.
    Returns:
        pd.DataFrame: A DataFrame containing the processed NBA data.
    """
    games = get_schedule_games(date_to_predict)
    if games.empty:
        print("No games found for the specified date.")
        raise ValueError(
            "No games found for the specified date."
        )  # Return empty DataFrame if no games found

    # Extract just the date portion (first 10 chars: YYYY-MM-DD) and combine with time
    games["GAME_TIME"] = pd.to_datetime(
        games["GAME_DATE_EST"].astype(str).str[:10] + " " + games["GAME_STATUS_TEXT"],
        format="%Y-%m-%d %I:%M %p ET",
        errors="coerce",
    )
    # Make it timezone-aware (Eastern Time)
    games["GAME_TIME"] = games["GAME_TIME"].dt.tz_localize(
        "US/Eastern", ambiguous="infer", nonexistent="shift_forward"
    )

    seasons = get_last_two_nba_seasons(date_to_predict)

    # df_2, df_players_2 = load_all_nba_data(
    #     data_path + "/" + "season_games_data", seasons=seasons
    # )

    df, df_players = load_all_nba_data_from_db(seasons=seasons)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")

    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    # df.dropna(inplace=True)
    df.dropna(subset=["PTS"], inplace=True)

    # convert to string TEAM_ID
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

    df = standardize_and_merge_nba_data(df, games)

    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    df = merge_teams_df_with_odds(df_odds=df_odds, df_team=df)
    df["TOTAL_POINTS"] = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["DIFF_FROM_LINE"] = df["TOTAL_POINTS"] - df["TOTAL_OVER_UNDER_LINE"]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")

    valid_games = df["GAME_ID"].value_counts()
    valid_games = valid_games[
        valid_games == 2
    ].index  # Keep only games with exactly 2 rows

    # Filter dataset to keep only valid GAME_IDs
    df = df[df["GAME_ID"].isin(valid_games)]

    df.loc[:, "SEASON_TYPE"] = df["GAME_ID"].apply(classify_season_type)
    df.loc[:, "SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)

    df = add_last_season_playoff_games(df)

    # Compute the number of games played by each team in the regular season.

    df.sort_values(by="GAME_DATE", ascending=True, inplace=True)

    group_cols = ["SEASON_TYPE", "SEASON_ID", "TEAM_ID"]

    # Assign game numbers in chronological order (1 = first game, 82 = last game)
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

    ##### here add PLAYER INJURIES and STATS

    rows_new_game = create_df_players_new_game(games, df_players)

    df_players = pd.concat([df_players, rows_new_game], ignore_index=True)

    df_players.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    injury_report_df = retrieve_injury_report_as_df(
        nba_injury_reports_url, reports_path=reports_path
    )

    injury_dict, games_not_updated = process_injury_data(games, injury_report_df)

    stats = ["PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT"]
    # Add a row of the new game for players

    df = attach_top3_stats(
        df, df_players, injury_dict, stats, game_date_limit=date_to_predict
    )

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

    # Append '_BEFORE' to renamed columns if starting with 'TOP' (already done)
    # (Your original columns already include "_BEFORE", if required to add more explicitly, adjust as below)
    df.rename(
        columns={
            col: f"{col}_BEFORE"
            for col in df.columns
            if col.startswith("TOP") and not col.endswith("_BEFORE")
        },
        inplace=True,
    )

    # Merge home and away stats
    static_columns = [
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "SEASON_TYPE",
        "SEASON_YEAR",
        "IS_OVERTIME",
        "GAME_TIME",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
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
            "GAME_TIME",
            "SEASON_TYPE",
            "IS_PLAYOFF_GAME",
            "PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY",
            "PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME",
            "SEASON_YEAR",
        ]
    )
    # insert columns that have BEFORE in the name
    columns_info_before.extend([col for col in df_merged.columns if "BEFORE" in col])

    odds_columns = [
        "TOTAL_OVER_UNDER_LINE",
        "SPREAD",
        "MONEYLINE_TEAM_HOME",
        "MONEYLINE_TEAM_AWAY",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]

    columns_info_before.extend(odds_columns)

    df_to_predict = df_merged[columns_info_before + ["TOTAL_POINTS"]].copy()
    df_to_predict = df_to_predict[df_to_predict["SEASON_TYPE"] != "Preseason"]
    df_to_predict["TOTAL_PTS_SEASON_BEFORE_AVG"] = (
        df_to_predict["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        + df_to_predict["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    )

    df_to_predict["TOTAL_PTS_LAST_GAMES_AVG"] = (
        df_to_predict["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        + df_to_predict["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )

    df_to_predict["BACK_TO_BACK"] = (
        df_to_predict["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"] == 1
    ) & (df_to_predict["REST_DAYS_BEFORE_MATCH_TEAM_HOME"] == 1)

    df_to_predict["DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE_MATCH"] = (
        df_to_predict["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        - df_to_predict["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )
    df_to_predict["DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE_MATCH"] = (
        df_to_predict["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
        - df_to_predict["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
    )

    df_to_predict.loc[df_to_predict["MATCHUP_TEAM_HOME"] == 0, "MATCHUP_TEAM_HOME"] = (
        df_to_predict["TEAM_ABBREVIATION_TEAM_HOME"]
        + " vs. "
        + df_to_predict["TEAM_ABBREVIATION_TEAM_AWAY"]
    )

    df_to_predict.loc[df_to_predict["MATCHUP_TEAM_AWAY"] == 0, "MATCHUP_TEAM_AWAY"] = (
        df_to_predict["TEAM_ABBREVIATION_TEAM_AWAY"]
        + " @ "
        + df_to_predict["TEAM_ABBREVIATION_TEAM_HOME"]
    )
    # filter for date_to_predict
    if isinstance(date_to_predict, str):
        date_to_predict = pd.to_datetime(date_to_predict, format="%Y-%m-%d")

    if filter_for_date_to_predict:
        df_to_predict = df_to_predict[df_to_predict["GAME_DATE"] == date_to_predict]
        assert (
            df_to_predict["TOTAL_POINTS"] == 0
        ).all(), "Error: TOTAL_POINTS contains non-zero values!"
        df_to_predict.drop(columns=["TOTAL_POINTS"], inplace=True)
    
    df_to_predict.drop(columns=["IS_OVERTIME"], inplace=True)

    # df_to_predict['TOTAL_OVER_UNDER_LINE'] = None
    first_col = "TOTAL_OVER_UNDER_LINE"
    df_to_predict = df_to_predict[
        [first_col] + [col for col in df_to_predict.columns if col != first_col]
    ]

    # add INJURIES NOT YET SUBMITTED to games games_not_updated in TOTAL_OVER_UNDER_LINE
    if games_not_updated is not None:
        for game_id in games_not_updated:
            df_to_predict.loc[
                df_to_predict["GAME_ID"] == game_id, "TOTAL_OVER_UNDER_LINE"
            ] = "INJURIES NOT YET SUBMITTED"

    print()
    print("--" * 20)
    print(f"Processed data for {date_to_predict.date()}")
    print(f"Number of games: {df_to_predict.shape[0]}")
    print("--" * 20)
    print()
    return df_to_predict


if __name__ == "__main__":
    # Constants
    DATA_PATH = "/home/adrian_alvarez/Projects/NBA-predictor/data/"
    OUTPUT_PATH = "/home/adrian_alvarez/Projects/NBA-predictor/injury_reports/"
    NBA_INJURY_REPORTS_URL = (
        "https://official.nba.com/nba-injury-report-2024-25-season/"
    )

    # get today
    date_to_predict = datetime.now().strftime("%Y-%m-%d")
    df_to_predict = create_df_to_predict(
        DATA_PATH, date_to_predict, NBA_INJURY_REPORTS_URL, OUTPUT_PATH
    )
    df_to_predict
