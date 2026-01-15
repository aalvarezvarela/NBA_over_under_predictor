"""
NBA Over/Under Predictor - Referee Data Processing Module

This module handles loading and processing referee data for NBA games,
including computing referee-specific features based on historical performance.
"""

import pandas as pd
from tqdm import tqdm


def load_refs_data_from_db(seasons):
    """
    Load referee data from database for the specified seasons.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])

    Returns:
        pd.DataFrame: Combined referee data for all seasons
    """
    from postgre_DB.db_config import connect_nba_db, get_schema_name_refs
    from psycopg import sql

    schema = get_schema_name_refs()
    table = "nba_refs"  # table name in refs schema

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
        df_refs = pd.read_sql_query(query, conn, params=(season_years,))

        # Convert column names to uppercase to match expected format
        df_refs.columns = df_refs.columns.str.upper()

        # Ensure GAME_ID is string for consistent merging
        if "GAME_ID" in df_refs.columns:
            df_refs["GAME_ID"] = df_refs["GAME_ID"].astype(str)

        # Remove duplicates
        df_refs = df_refs.drop_duplicates()

        print(f"Loaded {len(df_refs)} referee records from database")
        return df_refs

    except Exception as e:
        print(f"Error loading referee data from database: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    finally:
        if conn is not None:
            conn.close()


def compute_referee_features(df_refs_pivot):
    """
    Compute referee-specific features for each game based on historical performance.

    For each referee (REF_1, REF_2, REF_3) in each game, calculates:
    1. The difference between:
       - Average TOTAL_POINTS in games where that referee participated (as any of REF_1, REF_2, REF_3)
       - Average TOTAL_POINTS in games where that referee did NOT participate
    2. The same calculation using DIFFERENCE_FROM_LINE instead of TOTAL_POINTS

    Constraints:
    - Only uses data from the last two seasons (current season + previous season)
    - Only uses past games (games before the current game's GAME_DATE)
    - Never includes the current game in any calculation (prevents data leakage)
    - Games are ordered by GAME_DATE

    Args:
        df_refs_pivot (pd.DataFrame): DataFrame with columns:
            - GAME_ID: Unique game identifier
            - GAME_DATE: Date of the game
            - SEASON_YEAR: Year of the season
            - TOTAL_POINTS: Total points scored in the game
            - TOTAL_OVER_UNDER_LINE: Over/under line for the game
            - REF_1, REF_2, REF_3: Names of the three referees
            - DIFFERENCE_FROM_LINE: TOTAL_POINTS - TOTAL_OVER_UNDER_LINE

    Returns:
        pd.DataFrame: Original DataFrame with additional columns:
            - REF_1_TOTAL_POINTS_DIFF: Difference in avg total points with/without REF_1
            - REF_2_TOTAL_POINTS_DIFF: Difference in avg total points with/without REF_2
            - REF_3_TOTAL_POINTS_DIFF: Difference in avg total points with/without REF_3
            - REF_1_DIFF_FROM_LINE_DIFF: Difference in avg diff from line with/without REF_1
            - REF_2_DIFF_FROM_LINE_DIFF: Difference in avg diff from line with/without REF_2
            - REF_3_DIFF_FROM_LINE_DIFF: Difference in avg diff from line with/without REF_3
            - REF_TRIO_TOTAL_POINTS_DIFF: Difference in avg total points when all 3 refs appear together
            - REF_TRIO_DIFF_FROM_LINE_DIFF: Difference in avg diff from line when all 3 refs appear together
    """
    # Ensure GAME_DATE is datetime
    df = df_refs_pivot.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort by GAME_DATE to ensure chronological order
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Initialize new feature columns
    for ref_num in [1, 2, 3]:
        df[f"REF_{ref_num}_TOTAL_POINTS_DIFF"] = 0.0
        df[f"REF_{ref_num}_DIFF_FROM_LINE_DIFF"] = 0.0

    # Initialize trio features
    df["REF_TRIO_TOTAL_POINTS_DIFF"] = 0.0
    df["REF_TRIO_DIFF_FROM_LINE_DIFF"] = 0.0

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

        # Process each referee in the current game
        for ref_num in [1, 2, 3]:
            ref_col = f"REF_{ref_num}"
            ref_name = current_game[ref_col]

            # Skip if referee name is missing
            if pd.isna(ref_name):
                continue

            # Identify games where this referee participated (in any position)
            ref_participated = (
                (past_games["REF_1"] == ref_name)
                | (past_games["REF_2"] == ref_name)
                | (past_games["REF_3"] == ref_name)
            )

            games_with_ref = past_games[ref_participated]
            games_without_ref = past_games[~ref_participated]

            # Calculate TOTAL_POINTS averages
            if len(games_with_ref) > 0 and len(games_without_ref) > 0:
                avg_points_with = games_with_ref["TOTAL_POINTS"].mean()
                avg_points_without = games_without_ref["TOTAL_POINTS"].mean()
                df.at[idx, f"REF_{ref_num}_TOTAL_POINTS_DIFF"] = (
                    avg_points_with - avg_points_without
                )

            # Calculate DIFFERENCE_FROM_LINE averages
            if len(games_with_ref) > 0 and len(games_without_ref) > 0:
                avg_diff_with = games_with_ref["DIFFERENCE_FROM_LINE"].mean()
                avg_diff_without = games_without_ref["DIFFERENCE_FROM_LINE"].mean()
                df.at[idx, f"REF_{ref_num}_DIFF_FROM_LINE_DIFF"] = (
                    avg_diff_with - avg_diff_without
                )

        # Process referee trio (all three referees together, regardless of order)
        ref_trio = set()
        for ref_num in [1, 2, 3]:
            ref_name = current_game[f"REF_{ref_num}"]
            if not pd.isna(ref_name):
                ref_trio.add(ref_name)

        # Only calculate trio features if we have all three referees
        if len(ref_trio) == 3:
            # Identify past games where this exact trio officiated together (in any order)
            trio_participated = past_games.apply(
                lambda row: set([row["REF_1"], row["REF_2"], row["REF_3"]]) == ref_trio
                if all(not pd.isna(row[f"REF_{i}"]) for i in [1, 2, 3])
                else False,
                axis=1,
            )

            games_with_trio = past_games[trio_participated]
            games_without_trio = past_games[~trio_participated]

            # Calculate TOTAL_POINTS averages for trio
            if len(games_with_trio) > 0 and len(games_without_trio) > 0:
                avg_points_with_trio = games_with_trio["TOTAL_POINTS"].mean()
                avg_points_without_trio = games_without_trio["TOTAL_POINTS"].mean()
                df.at[idx, "REF_TRIO_TOTAL_POINTS_DIFF"] = (
                    avg_points_with_trio - avg_points_without_trio
                )

            # Calculate DIFFERENCE_FROM_LINE averages for trio
            if len(games_with_trio) > 0 and len(games_without_trio) > 0:
                avg_diff_with_trio = games_with_trio["DIFFERENCE_FROM_LINE"].mean()
                avg_diff_without_trio = games_without_trio[
                    "DIFFERENCE_FROM_LINE"
                ].mean()
                df.at[idx, "REF_TRIO_DIFF_FROM_LINE_DIFF"] = (
                    avg_diff_with_trio - avg_diff_without_trio
                )

    return df


def process_referee_data_for_training(seasons, df_train):
    """
    Load referee data from database, transform it, merge with training data,
    and compute referee-specific features.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])
        df_train (pd.DataFrame): Training DataFrame with GAME_ID, GAME_DATE, SEASON_YEAR,
                                TOTAL_POINTS, and TOTAL_OVER_UNDER_LINE columns

    Returns:
        pd.DataFrame: DataFrame with referee features, or None if no referee data available
    """
    df_refs = load_refs_data_from_db(seasons)

    # Transform referee data to have one row per game with REF_1, REF_2, REF_3
    if not df_refs.empty and "GAME_ID" in df_refs.columns:
        # Create full name column
        df_refs["FULL_NAME"] = df_refs["FIRST_NAME"] + " " + df_refs["LAST_NAME"]

        # Ensure GAME_DATE is datetime in df_refs
        df_refs["GAME_DATE"] = pd.to_datetime(df_refs["GAME_DATE"])

        # Group by GAME_ID and create REF_1, REF_2, REF_3 columns
        df_refs_pivot = (
            df_refs.groupby("GAME_ID")
            .apply(
                lambda x: pd.Series(
                    {
                        "REF_1": x["FULL_NAME"].iloc[0] if len(x) > 0 else None,
                        "REF_2": x["FULL_NAME"].iloc[1] if len(x) > 1 else None,
                        "REF_3": x["FULL_NAME"].iloc[2] if len(x) > 2 else None,
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )

        # Ensure GAME_ID is string in both dataframes
        df_train["GAME_ID"] = df_train["GAME_ID"].astype(str)
        df_refs_pivot["GAME_ID"] = df_refs_pivot["GAME_ID"].astype(str)
        df_refs["GAME_ID"] = df_refs["GAME_ID"].astype(str)

        df_train_temp = df_train[
            [
                "GAME_ID",
                "GAME_DATE",
                "SEASON_YEAR",
                "TOTAL_POINTS",
                "TOTAL_OVER_UNDER_LINE",
            ]
        ].copy()

        # Join based on GAME_ID the df_refs_pivot and df_train_temp, keeping all columns
        df_refs_pivot = df_train_temp.merge(
            df_refs_pivot,
            on="GAME_ID",
            how="inner",
        )
        df_refs_pivot["DIFFERENCE_FROM_LINE"] = (
            df_refs_pivot["TOTAL_POINTS"] - df_refs_pivot["TOTAL_OVER_UNDER_LINE"]
        )

        # Compute referee features
        print("Computing referee features...")
        df_refs_pivot = compute_referee_features(df_refs_pivot)

        return df_refs_pivot

    else:
        print("No referee data available to merge")
        return None


def add_referee_features_to_training_data(seasons, df_train):
    """
    Add referee-specific features to the training DataFrame.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])
        df_train (pd.DataFrame): Training DataFrame with GAME_ID, GAME_DATE, SEASON_YEAR,
                                TOTAL_POINTS, and TOTAL_OVER_UNDER_LINE columns
    Returns:
        pd.DataFrame: Training DataFrame with added referee features    
    """
    df_refs_pivot = process_referee_data_for_training(seasons, df_train)

    # Merge referee features into training data
    if df_refs_pivot is not None:
        ref_feature_cols = [
            "GAME_ID",
            "REF_1_TOTAL_POINTS_DIFF",
            "REF_2_TOTAL_POINTS_DIFF",
            "REF_3_TOTAL_POINTS_DIFF",
            "REF_1_DIFF_FROM_LINE_DIFF",
            "REF_2_DIFF_FROM_LINE_DIFF",
            "REF_3_DIFF_FROM_LINE_DIFF",
            "REF_TRIO_TOTAL_POINTS_DIFF",
            "REF_TRIO_DIFF_FROM_LINE_DIFF",
        ]
        
        # Select only the feature columns from df_refs_pivot
        df_refs_features = df_refs_pivot[ref_feature_cols].copy()
        
        # Merge with df_train based on GAME_ID
        df_train = df_train.merge(df_refs_features, on="GAME_ID", how="left")
        
        print("\nReferee features successfully merged into training data!")
        print(f"Training data shape after merge: {df_train.shape}")
    return df_train