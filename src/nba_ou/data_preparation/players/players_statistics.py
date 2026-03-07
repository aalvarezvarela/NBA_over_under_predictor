import numpy as np
import pandas as pd

EWMA_HALFLIFE_GAMES = 10


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
        df_inj = df[df["GAME_DATE"] < date].sort_values(["PLAYER_ID", "GAME_DATE"])
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
    df_last.loc[:, "MEETS_MIN_THRESHOLD"] = (
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
    df_players: pd.DataFrame, stat_col: str = "PTS", ewm_halflife_games: int = EWMA_HALFLIFE_GAMES
) -> pd.DataFrame:
    """
    Recency-weighted average (EWMA) of `stat_col` per (SEASON_ID, PLAYER_ID),
    using only prior valid appearances (MIN > 0) and excluding the current game.

    - Excludes the current game's stat from the estimate (via shift).
    - Gives slightly higher weight to recent games.
    - If a player has no prior valid game, the value is 0.
    - Only considers games where MIN > 0 as valid appearances.
    - Groups by both SEASON_ID and PLAYER_ID.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        stat_col (str): Column name for the statistic to compute average for
        ewm_halflife_games (int): EWMA halflife in games (higher = less reactive)

    Returns:
        pd.DataFrame: Updated DataFrame with recency-weighted average columns
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

    # Keep only valid appearances for the signal; invalid games are ignored.
    valid_mask = out["MIN"].fillna(0) > 0
    valid_stat = out[stat_col].where(valid_mask)

    # Use only prior games (shift) to avoid target leakage.
    shifted_valid_stat = valid_stat.groupby([out["SEASON_ID"], out["PLAYER_ID"]]).shift(1)

    # Recency-weighted estimate by player-season.
    out[f"{stat_col}_CUM_AVG"] = (
        shifted_valid_stat.groupby(
            [out["SEASON_ID"], out["PLAYER_ID"]],
            group_keys=False,
        ).transform(
            lambda s: s.ewm(
                halflife=ewm_halflife_games,
                adjust=False,
                min_periods=1,
                ignore_na=True,
            ).mean()
        )
    ).fillna(0)

    return out
