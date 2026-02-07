"""
Module for merging odds data with games data based on game_id.

This provides an alternative to the date-based merge, using game_id for exact matching.
Only merges total line, spread, and moneyline - no additional columns.
"""

from typing import Literal

import numpy as np
import pandas as pd


def merge_total_spread_moneyline_by_game_id(
    df_odds: pd.DataFrame,
    df_team: pd.DataFrame,
    book: str = "bet365",
    total_lines_mode: Literal["selected", "all", "none"] = "all",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Merge team dataframe with odds data using game_id.

    total_lines_mode:
      - "none": do not merge any total line
      - "selected": merge TOTAL_OVER_UNDER_LINE from the selected `book` only
      - "all": merge total lines for all known books into TOTAL_LINE_<book> columns
    """
    if df_odds.empty:
        print("Warning: Odds dataframe is empty")
        return df_team

    if "game_id" not in df_odds.columns:
        print("Warning: 'game_id' column not found in odds dataframe")
        return df_team

    if "GAME_ID" not in df_team.columns:
        print("Warning: 'GAME_ID' column not found in team dataframe")
        return df_team

    # Debug: print df_team rows whose GAME_ID is not in df_odds
    if debug:
        required_team_cols = {"GAME_DATE", "TEAM_NAME"}
        missing_team_cols = [c for c in required_team_cols if c not in df_team.columns]
        if missing_team_cols:
            print(
                f"Debug warning: df_team missing columns needed for debug print: {missing_team_cols}"
            )
        else:
            games_df = df_team[
                ~df_team["SEASON_TYPE"].isin(["Preseason", "All Star"])
            ].copy()
            games_df = games_df[games_df["SEASON_YEAR"] >= 2019]
            team_game_ids = set(games_df["GAME_ID"].dropna().unique())
            odds_game_ids = set(df_odds["game_id"].dropna().unique())

            missing_game_ids = team_game_ids - odds_game_ids
            if missing_game_ids:
                df_missing = (
                    games_df.loc[
                        games_df["GAME_ID"].isin(missing_game_ids),
                        ["GAME_ID", "TEAM_NAME", "GAME_DATE"],
                    ]
                    .drop_duplicates()
                    .sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"])
                )
                print(
                    f"[DEBUG] Games in df_team but not in df_odds: "
                    f"{len(missing_game_ids)} GAME_IDs, {len(df_missing)} team-rows"
                )
                print(df_missing.to_string(index=False))
            else:
                print("[DEBUG] All df_team GAME_IDs are present in df_odds.")

    # Base columns for the selected book (spread + moneyline)
    spread_home_col = f"spread_{book}_line_home"
    spread_away_col = f"spread_{book}_line_away"
    ml_home_col = f"ml_{book}_price_home"
    ml_away_col = f"ml_{book}_price_away"

    required_cols = [
        "game_id",
        spread_home_col,
        spread_away_col,
        ml_home_col,
        ml_away_col,
    ]

    # Totals handling
    known_total_sources = [
        ("consensus_opener", "total_consensus_opener_line_over"),
        ("betmgm", "total_betmgm_line_over"),
        ("fanduel", "total_fanduel_line_over"),
        ("caesars", "total_caesars_line_over"),
        ("bet365", "total_bet365_line_over"),
        ("draftkings", "total_draftkings_line_over"),
        ("fanatics_sportsbook", "total_fanatics_sportsbook_line_over"),
    ]

    if total_lines_mode == "selected":
        total_selected_col = f"total_{book}_line_over"
        required_cols.append(total_selected_col)

    elif total_lines_mode == "all":
        required_cols.extend([col for _, col in known_total_sources])

    # Check required columns exist
    missing_cols = [c for c in required_cols if c not in df_odds.columns]
    if missing_cols:
        print(f"Warning: Missing columns in odds dataframe: {missing_cols}")
        raise ValueError(f"Missing columns in odds dataframe: {missing_cols}")
        # return df_team

    # Subset odds
    df_odds_subset = df_odds[required_cols].copy()

    # Rename
    rename_map = {
        "game_id": "GAME_ID",
        spread_home_col: "SPREAD_HOME",
        spread_away_col: "SPREAD_AWAY",
        ml_home_col: "MONEYLINE_HOME",
        ml_away_col: "MONEYLINE_AWAY",
    }

    
    rename_map[f"total_{book}_line_over"] = "TOTAL_OVER_UNDER_LINE"

    if total_lines_mode == "all":
        for src, col in known_total_sources:
            if src == book:
                continue  # already handled by selected mode
            rename_map[col] = f"TOTAL_LINE_{src}"

    df_odds_subset = df_odds_subset.rename(columns=rename_map)

    # Merge
    df_merged = df_team.merge(
        df_odds_subset,
        on="GAME_ID",
        how="left",
        suffixes=("", "_odds"),
    )

    # Vectorized assignment for SPREAD and MONEYLINE
    if "HOME" in df_merged.columns:
        home_mask = df_merged["HOME"].astype(bool)
        df_merged["SPREAD"] = np.where(
            home_mask, df_merged["SPREAD_HOME"], df_merged["SPREAD_AWAY"]
        )
        df_merged["MONEYLINE"] = np.where(
            home_mask, df_merged["MONEYLINE_HOME"], df_merged["MONEYLINE_AWAY"]
        )
    else:
        print("Warning: 'HOME' column not found in team dataframe")
        df_merged["SPREAD"] = df_merged["SPREAD_HOME"]
        df_merged["MONEYLINE"] = df_merged["MONEYLINE_HOME"]

    # Drop temp columns
    df_merged = df_merged.drop(
        columns=["SPREAD_HOME", "SPREAD_AWAY", "MONEYLINE_HOME", "MONEYLINE_AWAY"],
        errors="ignore",
    )

    print(
        f"Merged {len(df_merged)} rows with odds data from {book} (total_lines_mode={total_lines_mode})"
    )
    return df_merged


def merge_remaining_odds_by_game_id(
    df_odds: pd.DataFrame,
    df_merged: pd.DataFrame,
    exclude_books: list[str] | None = ["bet365"],
) -> pd.DataFrame:
    """
    Merge all remaining odds data with merged home/away dataframe using game_id.

    This function should be called AFTER merge_home_away_data(), which creates one row
    per game. It merges all odds columns except those already merged by
    merge_total_spread_moneyline_by_game_id().

    Args:
        df_odds: Merged odds dataframe (from Yahoo + Sportsbook merge)
        df_merged: Merged home/away dataframe (one row per game) with GAME_ID column
        exclude_books: List of books to exclude specific columns for (default: ["bet365"])
            This will exclude total_X_line_over/under, spread_X_line_home/away,
            ml_X_price_home/away for each book in the list.

    Returns:
        pd.DataFrame: Merged dataframe with all remaining odds columns added
    """
    if isinstance(exclude_books, str):
        exclude_books = [exclude_books]

    if df_odds.empty:
        print("Warning: Odds dataframe is empty")
        return df_merged

    # Ensure game_id column exists in both dataframes
    if "game_id" not in df_odds.columns:
        print("Warning: 'game_id' column not found in odds dataframe")
        return df_merged

    if "GAME_ID" not in df_merged.columns:
        print("Warning: 'GAME_ID' column not found in merged dataframe")
        return df_merged

    # Build list of columns to exclude
    columns_to_exclude = set()
    for book in exclude_books:
        columns_to_exclude.update(
            [
                f"total_{book}_line_over",
                f"total_{book}_line_under",
                f"total_{book}_price_over",
                f"total_{book}_price_under",
                f"spread_{book}_line_home",
                f"spread_{book}_line_away",
                f"spread_{book}_price_home",
                f"spread_{book}_price_away",
                f"ml_{book}_price_home",
                f"ml_{book}_price_away",
            ]
        )

    # Also exclude metadata columns that are already in df_merged
    columns_to_exclude.update(
        [
            "game_date",
            "season_year",
            "team_home",
            "team_away",
        ]
    )

    # Select columns from odds to merge (all except excluded)
    cols_to_merge = ["game_id"]  # Always include game_id
    for col in df_odds.columns:
        if col not in columns_to_exclude and col != "game_id":
            cols_to_merge.append(col)

    # Check if there are any columns to merge besides game_id
    if len(cols_to_merge) <= 1:
        print("Warning: No remaining odds columns to merge after exclusions")
        return df_merged

    df_odds_subset = df_odds[cols_to_merge].copy()

    # Rename game_id to GAME_ID for merging
    df_odds_subset = df_odds_subset.rename(columns={"game_id": "GAME_ID"})

    # Merge with merged dataframe
    df_result = df_merged.merge(
        df_odds_subset,
        on="GAME_ID",
        how="left",
        suffixes=("", "_odds"),
    )

    print(
        f"Merged {len(cols_to_merge) - 1} remaining odds columns to {len(df_result)} rows"
    )

    return df_result
