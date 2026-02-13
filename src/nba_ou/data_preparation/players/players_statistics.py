import numpy as np
import pandas as pd


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
                       If there are no same-day rows (e.g., scheduled games),
                       fallback to each player's latest game prior to `date`.
                       If True, consider last game prior to `date`.
        lowest (bool): If False (default), return highest averages (descending).
                      If True, return lowest averages (ascending)
        n_players (int): Number of players to return
        min_minutes (int): Minimum average minutes threshold

    Returns:
        list: List of tuples (player_id, player_name, cumulative_average)
    """
    if stat_col == "DEF_RATING":
        lowest = True

    if injured:
        min_minutes = min_minutes * 0.8

    if df.empty:
        return []


    if injured:
        # For injured players: last game *before* `date`
        df_inj = df[df["GAME_DATE"] < date]
        df_last = df_inj.groupby("PLAYER_ID", as_index=False).tail(1).copy()
    
    else:
        # For non-injured players, keep existing behavior for historical rows.
        # For scheduled games (no same-day player boxscore yet), fallback to each
        # player's latest game before `date`.
        df_same_day = df[df["GAME_DATE"] == date].copy()
        if df_same_day.empty:
            df_prior = df[df["GAME_DATE"] < date]
            df_last = df_prior.groupby("PLAYER_ID", as_index=False).tail(1).copy()
        else:
            df_last = df_same_day

    if df_last.empty:
        return []

    # Check if MIN_CUM_AVG already exists (e.g., when stat_col="MIN")
    if "MIN_CUM_AVG" not in df_last.columns:
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
    df_last["MEETS_MIN_THRESHOLD"] = (
        df_last["MIN_CUM_AVG"].fillna(0) >= min_minutes
    ).astype(int)

    # Sort by the cumulative average column
    df_sorted = df_last.sort_values(
        by=["MEETS_MIN_THRESHOLD", cum_col], ascending=[False, lowest]
    )

    # Extract the top n (or bottom n) players
    chosen = df_sorted.head(n_players)

    top_or_bottom_n = list(
        zip(chosen["PLAYER_ID"], chosen["PLAYER_NAME"], chosen[cum_col], strict=True)
    )

    return top_or_bottom_n


def precompute_cumulative_avg_stat(
    df_players: pd.DataFrame, stat_col: str = "PTS"
) -> pd.DataFrame:
    """
    Cumulative average of `stat_col` per (SEASON_ID, PLAYER_ID) using only prior valid appearances (MIN > 0),
    excluding the current game.

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
    out = df_players.copy()

    # Defensive type conversions
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    out[stat_col] = pd.to_numeric(out[stat_col], errors="coerce")
    out["MIN"] = pd.to_numeric(out["MIN"], errors="coerce")

    # Sort to make shift/expanding meaningful
    out.sort_values(
        ["SEASON_ID", "PLAYER_ID", "GAME_DATE"], ascending=True, inplace=True
    )

    # Valid appearance (player actually played)
    out["VALID_GAME"] = (out["MIN"].fillna(0) > 0).astype(int)

    # Shifted values (prior game)
    shifted_stat = out.groupby(["SEASON_ID", "PLAYER_ID"])[stat_col].shift(1)
    valid_prev = (
        out.groupby(["SEASON_ID", "PLAYER_ID"])["VALID_GAME"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # Only count stat from prior games where the prior game was valid
    shifted_stat_valid = shifted_stat.fillna(0) * valid_prev

    # Cumsums over prior valid games
    out[f"CUMSUM_{stat_col}"] = shifted_stat_valid.groupby(
        [out["SEASON_ID"], out["PLAYER_ID"]]
    ).cumsum()

    out["GAME_COUNT"] = (
        valid_prev.groupby([out["SEASON_ID"], out["PLAYER_ID"]]).cumsum().astype(int)
    )

    # Compute average, avoiding divide-by-zero
    denom = out["GAME_COUNT"].replace(0, np.nan)
    out[f"{stat_col}_CUM_AVG"] = (out[f"CUMSUM_{stat_col}"] / denom).fillna(0)

    # Drop helper columns
    out.drop(columns=["VALID_GAME"], inplace=True)

    return out
