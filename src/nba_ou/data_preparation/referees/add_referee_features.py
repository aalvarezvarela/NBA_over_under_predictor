"""
NBA Over/Under Predictor - Referee Data Processing Module

This module handles loading and processing referee data for NBA games,
including computing referee-specific features based on historical performance.
"""

import re

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import resolve_main_total_line_col
from nba_ou.postgre_db.injuries_refs.fetch_refs_db.get_refs_db import (
    get_refs_data_from_db,
)
from tqdm import tqdm

# Metrics to compute for referee impact
REFEREE_METRICS = [
    "TOTAL_POINTS",  # Total points scored in the game
    "DIFF_FROM_LINE",  # Difference from over/under line
    "TOTAL_PF",  # Personal fouls called in the game
]


def _canonicalize_referee_name(name: str) -> str:
    """Normalize referee names so scheduled and historical sources match."""
    if pd.isna(name):
        return pd.NA
    name = str(name).strip()
    name = re.sub(r"\s*\(#\d+\)", "", name)  # remove assignment-site jersey suffix
    name = name.replace(".", "")
    name = re.sub(r"\s+", " ", name)
    return name or pd.NA


def _normalize_referee_slots(refs) -> pd.Series:
    """
    Convert a referee crew into deterministic REF_1/REF_2/REF_3 slots.

    The feature computation is order-invariant, so both scheduled and
    historical sources should be normalized to the same unique, sorted crew.
    """
    unique_refs = []
    seen = set()

    for ref_name in refs:
        normalized_name = _canonicalize_referee_name(ref_name)
        if pd.isna(normalized_name) or normalized_name in seen:
            continue
        seen.add(normalized_name)
        unique_refs.append(normalized_name)

    unique_refs.sort()

    return pd.Series(
        {
            "REF_1": unique_refs[0] if len(unique_refs) > 0 else pd.NA,
            "REF_2": unique_refs[1] if len(unique_refs) > 1 else pd.NA,
            "REF_3": unique_refs[2] if len(unique_refs) > 2 else pd.NA,
        }
    )


def compute_referee_features(df_refs_pivot):
    """
    Compute aggregate referee features for each game based on historical performance.

    For the current game's three referees, referee-position is ignored:
    1. For each referee, compute the metric delta:
       mean(metric in games with referee) - mean(metric in games without referee)
    2. Aggregate those per-referee deltas into:
       - mean across current referees
       - standard deviation across current referees
    3. Compute order-invariant trio features:
       - trio delta: mean(metric in games with exact trio) - mean(metric in games without trio)
       - trio std: standard deviation of metric in games with exact trio

    Args:
        df_refs_pivot (pd.DataFrame): DataFrame with columns:
            - GAME_ID: Unique game identifier
            - GAME_DATE: Date of the game
            - SEASON_YEAR: Year of the season
            - TOTAL_POINTS: Total points scored in the game
            - TOTAL_LINE_<main_book>: Main over/under line for the game
            - PF: Personal fouls called in the game
            - REF_1, REF_2, REF_3: Names of the three referees
            - DIFF_FROM_LINE: TOTAL_POINTS - TOTAL_LINE_<main_book>

    Returns:
        pd.DataFrame: Original DataFrame with additional columns:
            - REF_AVG_<METRIC>_DIFF_BEFORE
            - REF_STD_<METRIC>_DIFF_BEFORE
            - REF_TRIO_<METRIC>_DIFF_BEFORE
            - REF_TRIO_<METRIC>_STD_BEFORE
    """

    def _extract_unique_refs(row):
        return _normalize_referee_slots(
            [row["REF_1"], row["REF_2"], row["REF_3"]]
        ).dropna().tolist()

    # Ensure GAME_DATE is datetime
    df = df_refs_pivot.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort by GAME_DATE to ensure chronological order
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Cache order-invariant trio key per game for faster matching
    df["REF_TRIO_KEY"] = df.apply(
        lambda row: frozenset(_extract_unique_refs(row)), axis=1
    )

    # Initialize aggregate referee and trio features
    for metric in REFEREE_METRICS:
        df[f"REF_AVG_{metric}_DIFF_BEFORE"] = np.nan
        df[f"REF_STD_{metric}_DIFF_BEFORE"] = np.nan
        df[f"REF_TRIO_{metric}_DIFF_BEFORE"] = np.nan
        df[f"REF_TRIO_{metric}_STD_BEFORE"] = np.nan

    # Process each game
    for idx in tqdm(range(len(df)), desc="Computing referee features"):
        current_game = df.iloc[idx]
        current_date = current_game["GAME_DATE"]
        current_season = current_game["SEASON_YEAR"]
        # Define the two-season window (current season and previous season)
        target_seasons = [current_season, current_season - 1]

        # Get all past games in the two-season window (excluding current game)
        past_games = df[
            (df["GAME_DATE"] < current_date) & (df["SEASON_YEAR"].isin(target_seasons))
        ].copy()

        # Skip if no past games available
        if past_games.empty:
            continue

        current_refs = _extract_unique_refs(current_game)

        # Compute per-ref deltas and aggregate them into mean/std (position-agnostic)
        for metric in REFEREE_METRICS:
            per_ref_diffs = []
            for ref_name in current_refs:
                ref_participated = (
                    (past_games["REF_1"] == ref_name)
                    | (past_games["REF_2"] == ref_name)
                    | (past_games["REF_3"] == ref_name)
                )
                games_with_ref = past_games[ref_participated]
                games_without_ref = past_games[~ref_participated]

                if len(games_with_ref) > 0 and len(games_without_ref) > 0:
                    per_ref_diffs.append(
                        games_with_ref[metric].mean() - games_without_ref[metric].mean()
                    )

            if per_ref_diffs:
                per_ref_diffs_series = pd.Series(per_ref_diffs, dtype="float64")
                df.at[idx, f"REF_AVG_{metric}_DIFF_BEFORE"] = (
                    per_ref_diffs_series.mean()
                )
                df.at[idx, f"REF_STD_{metric}_DIFF_BEFORE"] = per_ref_diffs_series.std(
                    ddof=0
                )

        # Process referee trio (all three referees together, regardless of order)
        if len(current_refs) == 3:
            current_trio_key = frozenset(current_refs)
            trio_participated = past_games["REF_TRIO_KEY"] == current_trio_key
            games_with_trio = past_games[trio_participated]
            games_without_trio = past_games[~trio_participated]

            for metric in REFEREE_METRICS:
                if len(games_with_trio) > 0 and len(games_without_trio) > 0:
                    avg_with_trio = games_with_trio[metric].mean()
                    avg_without_trio = games_without_trio[metric].mean()
                    df.at[idx, f"REF_TRIO_{metric}_DIFF_BEFORE"] = (
                        avg_with_trio - avg_without_trio
                    )
                if len(games_with_trio) > 0:
                    trio_std = games_with_trio[metric].std(ddof=0)
                    df.at[idx, f"REF_TRIO_{metric}_STD_BEFORE"] = trio_std

    df = df.drop(columns=["REF_TRIO_KEY"])
    return df


def process_referee_data_for_training(
    seasons, df_merged, df_referees_scheduled=None, extra_game_ids=None
):
    """
    Load referee data from database, transform it, merge with training data,
    and compute referee-specific features.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])
        df_merged (pd.DataFrame): Training DataFrame with GAME_ID, GAME_DATE, SEASON_YEAR,
                                TOTAL_POINTS, and TOTAL_LINE_<main_book> columns
        new_ref_data (pd.DataFrame, optional): New referee data from scheduled games
    Returns:
        pd.DataFrame: DataFrame with referee features, or None if no referee data available
    """
    df_refs = get_refs_data_from_db(seasons, extra_game_ids=extra_game_ids)

    # Transform referee data to have one row per game with REF_1, REF_2, REF_3
    if not df_refs.empty and "GAME_ID" in df_refs.columns:
        # Create full name column
        df_refs["FULL_NAME"] = (df_refs["FIRST_NAME"] + " " + df_refs["LAST_NAME"]).map(
            _canonicalize_referee_name
        )

        # Ensure GAME_DATE is datetime in df_refs
        df_refs["GAME_DATE"] = pd.to_datetime(df_refs["GAME_DATE"])

        # Group by GAME_ID and create REF_1, REF_2, REF_3 columns
        df_refs_pivot = (
            df_refs.groupby("GAME_ID")
            .apply(lambda x: _normalize_referee_slots(x["FULL_NAME"]), include_groups=False)
            .reset_index()
        )

        if df_referees_scheduled is not None:
            # Append new referee assignments from scheduled games
            # Select only GAME_ID and REF columns from new_ref_data
            new_ref_data_subset = df_referees_scheduled[
                ["GAME_ID", "REF_1", "REF_2", "REF_3"]
            ].copy()
            for ref_col in ["REF_1", "REF_2", "REF_3"]:
                new_ref_data_subset[ref_col] = new_ref_data_subset[ref_col].map(
                    _canonicalize_referee_name
                )
            new_ref_data_subset[["REF_1", "REF_2", "REF_3"]] = new_ref_data_subset[
                ["REF_1", "REF_2", "REF_3"]
            ].apply(lambda row: _normalize_referee_slots(row), axis=1)

            # Get all unique scheduled referees
            scheduled_refs = set()
            for ref_col in ["REF_1", "REF_2", "REF_3"]:
                refs = new_ref_data_subset[ref_col].dropna().unique()
                scheduled_refs.update(refs)

            # Get all unique historical referees from database
            historical_refs = set()
            for ref_col in ["REF_1", "REF_2", "REF_3"]:
                refs = df_refs_pivot[ref_col].dropna().unique()
                historical_refs.update(refs)

            # Check for referees without historical data
            unmatched_refs = scheduled_refs - historical_refs
            if unmatched_refs:
                unmatched_list = sorted(list(unmatched_refs))
                raise ValueError(
                    f"Scheduled referee(s) not found in historical data: {unmatched_list}. "
                    f"These referees have no prior games to compute features from."
                )

            # Append new referee data to df_refs_pivot
            df_refs_pivot = pd.concat(
                [df_refs_pivot, new_ref_data_subset], ignore_index=True
            )

        # Ensure GAME_ID is string in both dataframes
        df_merged["GAME_ID"] = df_merged["GAME_ID"].astype(str)

        df_refs_pivot["GAME_ID"] = df_refs_pivot["GAME_ID"].astype(str)
        df_refs["GAME_ID"] = df_refs["GAME_ID"].astype(str)
        # Ensure exactly one crew row per game; prefer scheduled rows when provided.
        df_refs_pivot = df_refs_pivot.drop_duplicates(subset=["GAME_ID"], keep="last")

        main_total_line = resolve_main_total_line_col(df_merged)
        if main_total_line is None:
            raise ValueError(
                "No TOTAL_LINE_<book> column found in merged dataframe for referee features."
            )

        df_merged_temp = df_merged[
            [
                "GAME_ID",
                "GAME_DATE",
                "SEASON_YEAR",
                "TOTAL_POINTS",
                main_total_line,
                "TOTAL_PF",
            ]
        ].copy()

        # Join based on GAME_ID the df_refs_pivot and df_merged_temp, keeping all columns
        df_refs_pivot = df_merged_temp.merge(
            df_refs_pivot,
            on="GAME_ID",
            how="inner",
        )
        df_refs_pivot["DIFF_FROM_LINE"] = (
            df_refs_pivot["TOTAL_POINTS"] - df_refs_pivot[main_total_line]
        )

        # Compute referee features
        df_refs_pivot = compute_referee_features(df_refs_pivot)

        return df_refs_pivot

    else:
        print("No referee data available to merge")
        return None


def add_referee_features_to_training_data(
    seasons, df_merged, df_referees_scheduled=None, extra_game_ids=None
):
    """
    Add referee-specific features to the training DataFrame.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])
        df_merged (pd.DataFrame): Training DataFrame with GAME_ID, GAME_DATE, SEASON_YEAR,
                                TOTAL_POINTS, and TOTAL_LINE_<main_book> columns
    Returns:
        pd.DataFrame: Training DataFrame with added referee features
    """
    df_refs_pivot = process_referee_data_for_training(
        seasons,
        df_merged,
        df_referees_scheduled=df_referees_scheduled,
        extra_game_ids=extra_game_ids,
    )

    # Merge referee features into training data
    if df_refs_pivot is not None:
        # Dynamically build feature column list based on REFEREE_METRICS
        ref_feature_cols = ["GAME_ID"]

        # Add aggregate referee and trio features
        for metric in REFEREE_METRICS:
            ref_feature_cols.append(f"REF_AVG_{metric}_DIFF_BEFORE")
            ref_feature_cols.append(f"REF_STD_{metric}_DIFF_BEFORE")
            ref_feature_cols.append(f"REF_TRIO_{metric}_DIFF_BEFORE")
            ref_feature_cols.append(f"REF_TRIO_{metric}_STD_BEFORE")

        # Select only the feature columns from df_refs_pivot
        df_refs_features = df_refs_pivot[ref_feature_cols].copy()

        # Merge with df_merged based on GAME_ID
        df_merged = df_merged.merge(df_refs_features, on="GAME_ID", how="left")

        print("\nReferee features successfully merged into training data!")
        print(f"Training data shape after merge: {df_merged.shape}")

    return df_merged
