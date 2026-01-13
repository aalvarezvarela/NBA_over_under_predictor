"""
NBA Over/Under Predictor - Travel Processing Module

This module computes travel-related features for NBA teams, including:
- Distance traveled between consecutive games
- Rolling 7-day and 14-day travel distances
- Home vs away game logic

Uses geographic coordinates to calculate great-circle distances between cities.
"""

import numpy as np
import pandas as pd
from config.constants import CITY_TO_LATLON


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth using Haversine formula.

    Args:
        lat1 (float): Latitude of first point in degrees
        lon1 (float): Longitude of first point in degrees
        lat2 (float): Latitude of second point in degrees
        lon2 (float): Longitude of second point in degrees

    Returns:
        float: Distance in kilometers
    """
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def build_team_game_log(df):
    """
    Convert game-level dataframe into team-centric game log.

    Each game row is split into two rows (one for home team, one for away team).
    CITY represents where the game is PLAYED (not team's home city).

    Args:
        df (pd.DataFrame): Game-level dataframe with columns:
            - GAME_ID
            - GAME_DATE
            - TEAM_ID_TEAM_HOME
            - TEAM_CITY_TEAM_HOME
            - TEAM_ID_TEAM_AWAY
            - TEAM_CITY_TEAM_AWAY

    Returns:
        pd.DataFrame: Team-centric game log with columns:
            - GAME_ID
            - GAME_DATE
            - TEAM_ID
            - CITY (where game is played)
            - IS_HOME
    """
    # Home team: plays in their own city
    home = df[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID_TEAM_HOME",
            "TEAM_CITY_TEAM_HOME",
        ]
    ].copy()

    home.columns = ["GAME_ID", "GAME_DATE", "TEAM_ID", "CITY"]
    home["IS_HOME"] = True

    # Away team: plays in HOME TEAM'S city (travels TO opponent's city)
    away = df[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID_TEAM_AWAY",
            "TEAM_CITY_TEAM_HOME",  # Changed: away team plays in home team's city!
        ]
    ].copy()

    away.columns = ["GAME_ID", "GAME_DATE", "TEAM_ID", "CITY"]
    away["IS_HOME"] = False

    team_log = pd.concat([home, away], ignore_index=True)
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

    return team_log.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def add_travel_distance(team_log, city_coords=None):
    """
    Compute travel distance for each game based on previous game location.

    Rules:
    - If team name not in city_coords, distance = 0
    - If previous city not available (first game), distance = 0
    - If both current and previous games are home games, distance = 0
    - Otherwise, distance = great-circle distance between cities

    Args:
        team_log (pd.DataFrame): Team-centric game log with TEAM_ID, CITY, IS_HOME
        city_coords (dict, optional): Dictionary mapping city names to (lat, lon).
                                     Defaults to CITY_TO_LATLON from constants

    Returns:
        pd.DataFrame: Team log with TRAVEL_KM column added
    """
    if city_coords is None:
        city_coords = CITY_TO_LATLON

    team_log = team_log.copy()

    # Get previous game information
    team_log["PREV_CITY"] = team_log.groupby("TEAM_ID")["CITY"].shift(1)
    team_log["PREV_IS_HOME"] = team_log.groupby("TEAM_ID")["IS_HOME"].shift(1)

    def compute_distance(row):
        """Compute travel distance for a single game."""
        # First game for team or missing data
        if pd.isna(row["PREV_CITY"]) or pd.isna(row["CITY"]):
            return 0.0

        # Both games at home - no travel
        if row["IS_HOME"] and row["PREV_IS_HOME"]:
            return 0.0

        # Check if cities exist in coordinates dictionary
        if row["PREV_CITY"] not in city_coords or row["CITY"] not in city_coords:
            return 0.0

        # Calculate distance
        lat1, lon1 = city_coords[row["PREV_CITY"]]
        lat2, lon2 = city_coords[row["CITY"]]

        return haversine_km(lat1, lon1, lat2, lon2)

    team_log["TRAVEL_KM"] = team_log.apply(compute_distance, axis=1)

    return team_log


def add_rolling_distances(team_log):
    """
    Add rolling travel distance sums for multiple time windows.

    For each game, computes:
    - Total kilometers traveled in last 1 day
    - Total kilometers traveled in last 2 days
    - Total kilometers traveled in last 5 days
    - Total kilometers traveled in last 7 days
    - Total kilometers traveled in last 14 days

    Args:
        team_log (pd.DataFrame): Team log with GAME_DATE, TEAM_ID, and TRAVEL_KM

    Returns:
        pd.DataFrame: Team log with KM_LAST_*_DAYS columns for windows: 1, 2, 5, 7, 14
    """
    team_log = team_log.copy()

    # Sort by TEAM_ID and GAME_DATE before rolling operations
    team_log = team_log.sort_values(["TEAM_ID", "GAME_DATE"])
    team_log = team_log.set_index("GAME_DATE")

    # Define rolling windows
    windows = [1, 2, 5, 7, 14]

    for window in windows:
        col_name = f"KM_LAST_{window}_DAYS"
        team_log[col_name] = (
            team_log.groupby("TEAM_ID")["TRAVEL_KM"]
            .rolling(f"{window}D", closed="left")
            .sum()
            .reset_index(level=0, drop=True)
        )
        # Fill NaN values (first games with no history) with 0
        team_log[col_name] = team_log[col_name].fillna(0)

    return team_log.reset_index()


def merge_travel_features(df, team_log, log_scale=True):
    """
    Merge travel features back to original game-level dataframe.

    Creates columns for multiple rolling windows (1, 2, 5, 7, 14 days) for both home and away teams.

    Args:
        df (pd.DataFrame): Original game-level dataframe
        team_log (pd.DataFrame): Team log with travel features
        log_scale (bool): Whether to apply log transformation to travel features

    Returns:
        pd.DataFrame: Game-level dataframe with travel features added
    """
    # Define rolling windows
    windows = [1, 2, 5, 7, 14]

    # Build column lists dynamically
    km_cols = ["GAME_ID", "TEAM_ID"] + [f"KM_LAST_{w}_DAYS" for w in windows]

    # Filter team_log for home teams only and select relevant columns
    home_feats = team_log[team_log["IS_HOME"] == True][km_cols].copy()

    # Rename columns for home team
    rename_dict_home = {"TEAM_ID": "TEAM_ID_TEAM_HOME"}
    for w in windows:
        rename_dict_home[f"KM_LAST_{w}_DAYS"] = f"TOTAL_KM_IN_LAST_{w}_DAYS_HOME_TEAM"
    home_feats.rename(columns=rename_dict_home, inplace=True)

    # Filter team_log for away teams only and select relevant columns
    away_feats = team_log[team_log["IS_HOME"] == False][km_cols].copy()

    # Rename columns for away team
    rename_dict_away = {"TEAM_ID": "TEAM_ID_TEAM_AWAY"}
    for w in windows:
        rename_dict_away[f"KM_LAST_{w}_DAYS"] = f"TOTAL_KM_IN_LAST_{w}_DAYS_AWAY_TEAM"
    away_feats.rename(columns=rename_dict_away, inplace=True)

    # Merge home features
    df = df.merge(home_feats, on=["GAME_ID", "TEAM_ID_TEAM_HOME"], how="left")

    # Merge away features
    df = df.merge(away_feats, on=["GAME_ID", "TEAM_ID_TEAM_AWAY"], how="left")

    # Build list of all travel columns
    travel_columns = []
    for w in windows:
        travel_columns.append(f"TOTAL_KM_IN_LAST_{w}_DAYS_HOME_TEAM")
        travel_columns.append(f"TOTAL_KM_IN_LAST_{w}_DAYS_AWAY_TEAM")

    # Fill any remaining NaN values with 0 and optionally apply log transformation
    for col in travel_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            # Apply logarithmic transformation (log1p handles zeros gracefully)
            if log_scale:
                df[col] = np.log1p(df[col])

    return df


def compute_travel_features(df, log_scale=True):
    """
    Main function to compute all travel-related features for training data.

    This is the main entry point that orchestrates:
    1. Building team-centric game log
    2. Computing travel distances between consecutive games
    3. Calculating rolling sums for 1, 2, 5, 7, and 14 day windows
    4. Merging features back to original dataframe
    5. Optionally applying log transformation

    Args:
        df (pd.DataFrame): Game-level dataframe with required columns:
            - GAME_ID
            - GAME_DATE
            - TEAM_ID_TEAM_HOME
            - TEAM_CITY_TEAM_HOME
            - TEAM_ID_TEAM_AWAY
            - TEAM_CITY_TEAM_AWAY
        log_scale (bool): Whether to apply log1p transformation to travel distances

    Returns:
        pd.DataFrame: Original dataframe with added travel features for both home and away teams:
            - TOTAL_KM_IN_LAST_{1,2,5,7,14}_DAYS_HOME_TEAM
            - TOTAL_KM_IN_LAST_{1,2,5,7,14}_DAYS_AWAY_TEAM
    """
    print("Computing travel features...")

    # Step 1: Build team-centric game log
    team_log = build_team_game_log(df)

    # Step 2: Compute travel distances
    team_log = add_travel_distance(team_log)

    # Step 3: Compute rolling sums
    team_log = add_rolling_distances(team_log)

    # Step 4: Merge back to original dataframe
    df = merge_travel_features(df, team_log, log_scale=log_scale)

    print("Travel features computed successfully.")

    return df
