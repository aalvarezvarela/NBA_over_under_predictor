"""
NBA Over/Under Predictor - Training Data Creation Module

This module creates training datasets for NBA over/under prediction models.
It processes historical data from the last two seasons, computing all features
and statistics needed for model training, including injury data processing.
"""

import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.data_preparation.merged_home_away_data.add_features_after_merging import (
    add_derived_features_after_computed_stats,
    add_game_date_features,
    add_high_value_features_for_team_points,
)
from nba_ou.data_preparation.merged_home_away_data.merge_home_away import (
    merge_home_away_data,
)
from nba_ou.data_preparation.merged_home_away_data.select_train_columns import (
    select_training_columns,
)
from nba_ou.data_preparation.past_injuries.injury_effects import (
    add_top3_absence_effect_features_for_columns,
)
from nba_ou.data_preparation.players.attach_player_features import (
    add_player_history_features,
    clear_player_statistics,
)
from nba_ou.data_preparation.referees.add_referee_features import (
    add_referee_features_to_training_data,
)
from nba_ou.data_preparation.scheduled_games.merge_scheduled_with_existing_data import (
    standardize_and_merge_scheduled_games_to_team_data,
)
from nba_ou.data_preparation.team.cleaning_teams import adjust_overtime, clean_team_data
from nba_ou.data_preparation.team.filters import filter_valid_games
from nba_ou.data_preparation.team.merge_teams_df_with_odds import (
    merge_teams_df_with_odds,
)
from nba_ou.data_preparation.team.records import (
    add_last_season_playoff_games,
    add_team_record_before_game,
    compute_rest_days_before_match,
)
from nba_ou.data_preparation.team.rolling import compute_all_rolling_statistics
from nba_ou.data_preparation.team.totals import compute_total_points_features
from nba_ou.data_preparation.travel.travel_processing import compute_travel_features
from nba_ou.postgre_db import load_all_nba_data_from_db
from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (
    get_injury_data_from_db,
)
from nba_ou.postgre_db.odds.update_odds.upload_to_odds_db import load_odds_data
from nba_ou.utils.filter_by_date_range import filter_by_date_range
from nba_ou.utils.seasons import get_seasons_between_dates

warnings.simplefilter(action="ignore", category=FutureWarning)


def process_team_statistics_for_training(df, df_odds, scheduled_games=None):
    """
    Process and compute team statistics for training.

    This function handles:
    - Data cleaning and overtime adjustments
    - Merging game and odds data
    - Computing team records, win/loss statistics
    - Calculating rolling statistics and trends

    Args:
        df (pd.DataFrame): Team game statistics DataFrame
        df_odds (pd.DataFrame): Betting odds DataFrame
        scheduled_games (pd.DataFrame, optional): Scheduled games DataFrame
    Returns:
        pd.DataFrame: Processed team DataFrame
    """
    df = clean_team_data(df)
    df = adjust_overtime(df)

    if scheduled_games is not None:
        df = standardize_and_merge_scheduled_games_to_team_data(df, scheduled_games)

    df = merge_teams_df_with_odds(df_odds=df_odds, df_team=df)
    df = compute_total_points_features(df)
    df = filter_valid_games(df)

    df = add_last_season_playoff_games(df)

    df = add_team_record_before_game(df)
    df = compute_rest_days_before_match(df)

    # Compute all rolling statistics
    df = compute_all_rolling_statistics(df)

    df.loc[df["TOTAL_OVER_UNDER_LINE"] == 0, "TOTAL_OVER_UNDER_LINE"] = np.nan

    df = df.drop_duplicates(keep="first")

    return df


def process_player_statistics_for_training(
    df_players,
    df_team,
    df_injuries,
    older_limit_to_include,
    recent_limit_to_include,
    injury_dict_scheduled=None,
):
    """
    Process player statistics and prepare for training.

    This function handles:
    - Merging player data with game dates from team data
    - Converting player minutes from MM:SS format to decimal
    - Cleaning and deduplicating player data

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        df (pd.DataFrame): Processed team DataFrame with GAME_ID, GAME_DATE, SEASON_ID
        injury_dict_scheduled (dict, optional): Dictionary of scheduled injury data

    Returns:
        pd.DataFrame: Processed player DataFrame
    """

    df_players = clear_player_statistics(df_players, df_team)
    df_players = filter_by_date_range(
        df_players, older_limit_to_include, recent_limit_to_include
    )
    # Define statistics to compute for top players
    stats = ["PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT", "MIN"]

    # Attach top player statistics including injury data
    df, injured_dict = add_player_history_features(
        df_team,
        df_players,
        df_injuries,
        stats,
        injury_dict_scheduled=injury_dict_scheduled,
    )

    return df, injured_dict


def create_df_to_predict(
    todays_prediction: bool = False,
    scheduled_games: pd.DataFrame = None,
    df_referees_scheduled: pd.DataFrame = None,
    injury_dict_scheduled: dict = None,
    df_odds_scheduled: pd.DataFrame = None,
    recent_limit_to_include: str = None,
    older_limit_to_include: str = None,
) -> pd.DataFrame:
    """
    Create prediction dataset for NBA over/under prediction models.

    This function:
    - Loads data from database for seasons from older_limit_to_include (or 2006-07) to recent_limit_to_include
    - Processes injuries from database (not from live reports)
    - Computes all team and player statistics
    - Calculates rolling averages and trends
    - Merges home/away data and prepares final training features

    Args:
        recent_limit_to_include (str | datetime): Latest date to include in training data (YYYY-MM-DD)
        df_odds_scheduled (pd.DataFrame, optional): Scheduled betting odds data to include
        older_limit_to_include (str | datetime, optional): Starting date for training data. If None, starts from 2006-07

    Returns:
        pd.DataFrame: Complete training dataset with all features
    """
    if todays_prediction:
        assert (
            (df_referees_scheduled is not None)
            and (scheduled_games is not None)
            and (df_odds_scheduled is not None)
            and (injury_dict_scheduled is not None)
        ), "Scheduled games and referees data must be provided to include current day"

    # Set default recent_limit_to_include to yesterday if not provided
    if recent_limit_to_include is None:
        recent_limit_to_include = pd.Timestamp.now(
            tz=ZoneInfo("US/Eastern")
        ) - pd.Timedelta(days=1)

    recent_limit_to_include = pd.to_datetime(recent_limit_to_include, format="%Y-%m-%d")

    # Determine older_limit_to_include based on todays_prediction flag
    if todays_prediction:
        # If predicting today, go back two years from today
        older_limit_to_include = pd.Timestamp.now(
            tz=ZoneInfo("US/Eastern")
        ) - pd.Timedelta(days=365 * 2)
    elif older_limit_to_include is None:
        # Default to 2006 (start of 2006-07 season)
        older_limit_to_include = pd.to_datetime("2006-10-01")

    older_limit_to_include = pd.to_datetime(older_limit_to_include, format="%Y-%m-%d")

    # Determine which seasons to load
    seasons = get_seasons_between_dates(older_limit_to_include, recent_limit_to_include)

    df_odds = load_odds_data(season_years=seasons)

    if df_odds_scheduled is not None:
        df_odds = pd.concat([df_odds, df_odds_scheduled], ignore_index=True)
        df_odds.sort_values(by="game_date", inplace=True, ascending=False)
        df_odds.reset_index(drop=True, inplace=True)

    # Load game and player data from database
    df, df_players = load_all_nba_data_from_db(seasons=seasons)

    # Ensure GAME_DATE column is pandas Timestamp for df (df_players doesn't have it yet)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Filter df by date range
    df = filter_by_date_range(df, older_limit_to_include, recent_limit_to_include)

    original_columns = df.columns.tolist()
    # Get today day to predict

    print(f"Loading data for seasons: {seasons}")

    # Process team statistics
    df = process_team_statistics_for_training(
        df, df_odds, scheduled_games=scheduled_games
    )
    # Load injury data from database
    df_injuries = get_injury_data_from_db(seasons)

    # Add Players Statistics

    df, injured_dict = process_player_statistics_for_training(
        df_players,
        df,
        df_injuries,
        older_limit_to_include,
        recent_limit_to_include,
        injury_dict_scheduled=injury_dict_scheduled,
    )

    df_merged = merge_home_away_data(df)

    df_merged = add_referee_features_to_training_data(
        seasons, df_merged, df_referees_scheduled=df_referees_scheduled
    )

    df_training = select_training_columns(df_merged, original_columns)
    df_training = add_derived_features_after_computed_stats(df_training)

    df_training = add_top3_absence_effect_features_for_columns(
        df_training,
        injured_dict,
        home_player_cols=(
            "TOP1_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
            "TOP2_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
            "TOP3_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
            "TOP1_PLAYER_ID_MIN_BEFORE_TEAM_HOME",
            "TOP2_PLAYER_ID_MIN_BEFORE_TEAM_HOME",
        ),
        away_player_cols=(
            "TOP1_PLAYER_ID_PTS_BEFORE_TEAM_AWAY",
            "TOP2_PLAYER_ID_PTS_BEFORE_TEAM_AWAY",
            "TOP3_PLAYER_ID_PTS_BEFORE_TEAM_AWAY",
            "TOP1_PLAYER_ID_MIN_BEFORE_TEAM_AWAY",
            "TOP2_PLAYER_ID_MIN_BEFORE_TEAM_AWAY",
        ),
        out_prefix="TOP3_ABSENCE_EFFECT",
    )

    df_training = add_top3_absence_effect_features_for_columns(
        df_training,
        injured_dict,
        home_player_cols=(
            "TOP1_INJURED_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
            "TOP2_INJURED_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
            "TOP3_INJURED_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
            "TOP1_INJURED_PLAYER_ID_MIN_BEFORE_TEAM_HOME",
            "TOP2_INJURED_PLAYER_ID_MIN_BEFORE_TEAM_HOME",
        ),
        away_player_cols=(
            "TOP1_INJURED_PLAYER_ID_PTS_BEFORE_TEAM_AWAY",
            "TOP2_INJURED_PLAYER_ID_PTS_BEFORE_TEAM_AWAY",
            "TOP3_INJURED_PLAYER_ID_PTS_BEFORE_TEAM_AWAY",
            "TOP1_INJURED_PLAYER_ID_MIN_BEFORE_TEAM_AWAY",
            "TOP2_INJURED_PLAYER_ID_MIN_BEFORE_TEAM_AWAY",
        ),
        out_prefix="TOP3_INJURED_ABSENCE_EFFECT",
    )

    df_training = compute_travel_features(df_training, log_scale=True)
    df_training = add_high_value_features_for_team_points(df_training)
    df_training = add_game_date_features(df_training)

    print()
    print("--" * 20)
    print(f"Training data created up to {recent_limit_to_include}")
    print(f"Number of games: {df_training.shape[0]}")
    print(f"Number of features: {df_training.shape[1]}")
    print("--" * 20)
    print()

    return df_training


if __name__ == "__main__":
    output_path = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data"
    )
    # Create training data up to a specific date
    date_to_train = "2026-01-10"
    # older_limit_to_include = "2025-11-01"  # Optional: specify start date
    older_limit_to_include = "2023-12-01"  # Optional: specify start date

    # Get all info for scheduled games
    date_to_predict = pd.Timestamp.now(tz=ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")
    scheduled_games, df_referees_scheduled, injury_dict_scheduled, df_odds_scheduled = (
        get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            save_reports_path=SETTINGS.report_path,
            odds_api_key=SETTINGS.odds_api_key,
            odds_base_url=SETTINGS.odds_base_url,
        )
    )

    df_train = create_df_to_predict(
        todays_prediction=False,
        scheduled_games=scheduled_games,
        df_referees_scheduled=df_referees_scheduled,
        injury_dict_scheduled=injury_dict_scheduled,
        df_odds_scheduled=df_odds_scheduled,
        recent_limit_to_include=date_to_train,
        older_limit_to_include=older_limit_to_include,
    )

    output_name_before_referee = f"{output_path}/training_data{pd.to_datetime(date_to_train).strftime('%Y%m%d')}.csv"
    # df_train.to_csv(output_name_before_referee, index=False)
    print(f"Training data features saved to {output_name_before_referee}")
