"""
NBA Over/Under Predictor - Statistical Analysis Module

This module contains functions for computing rolling statistics, cumulative averages,
trend analysis, and other statistical computations on NBA game and player data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def classify_season_type(game_id):
    if game_id.startswith("001"):
        return "Preseason"
    elif game_id.startswith("002"):
        return "Regular Season"
    elif game_id.startswith("003"):
        return "All Star"
    elif game_id.startswith("004"):
        return "Playoffs"
    elif game_id.startswith("005"):
        return "Playoffs"  # in reality, this should be "Play-In Tournament", but for simplicity we'll use "Playoffs"
    elif game_id.startswith("006"):
        return "In-Season Final Game"
    return "Unknown"


def compute_rolling_stats(
    df: pd.DataFrame,
    param: str = "PTS",
    window: int = 5,
    season_avg: bool = False,
) -> pd.DataFrame:
    """
    Computes rolling averages for a given `param`, excluding the current row's game.

    Creates:
      - f"{param}_LAST_ALL_{window}_MATCHES_BEFORE"
      - f"{param}_LAST_HOME_AWAY_{window}_MATCHES_BEFORE"
      - f"{param}_SEASON_BEFORE_AVG" (if season_avg=True)

    Requirements:
      - Columns: TEAM_ID, HOME (bool), GAME_DATE, SEASON_YEAR, and `param`.
        (SEASON_YEAR is used to prevent cross-season contamination in rolling windows.)

    Notes:
      - Uses shift(1) everywhere to exclude the current game.
      - Computes rolling within (TEAM_ID, SEASON_YEAR) for "ANY"
      - Computes rolling within (TEAM_ID, SEASON_YEAR, HOME) for home/away split
      - Keeps final sort by GAME_DATE descending to match your pipeline style
    """
    if param not in df.columns:
        return df

    required_cols = {"TEAM_ID", "HOME", "GAME_DATE", "SEASON_YEAR"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_rolling_stats missing required columns: {sorted(missing)}"
        )

    out = df.copy()

    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    # Stable deterministic sort for rolling computations:
    # sort within team/season by date, and break ties by GAME_ID if present.
    sort_cols = ["TEAM_ID", "SEASON_YEAR", "GAME_DATE"]
    if "GAME_ID" in out.columns:
        sort_cols.append("GAME_ID")

    out.sort_values(sort_cols, ascending=True, inplace=True)

    last_n_avg_col = f"{param}_LAST_ALL_{window}_MATCHES_BEFORE"
    last_n_homeaway_col = f"{param}_LAST_HOME_AWAY_{window}_MATCHES_BEFORE"
    season_avg_col = f"{param}_SEASON_BEFORE_AVG"

    # Ensure numeric (keeps NaNs if coercion fails)
    series = pd.to_numeric(out[param], errors="coerce")

    # 1) Last N games (ANY) within season, excluding current game
    out[last_n_avg_col] = series.groupby(
        [out["TEAM_ID"], out["SEASON_YEAR"]]
    ).transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    # 2) Last N games split by HOME/AWAY within season, excluding current game
    out[last_n_homeaway_col] = series.groupby(
        [out["TEAM_ID"], out["SEASON_YEAR"], out["HOME"]]
    ).transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    # 3) Season-to-date average (within season and home/away), excluding current game
    if season_avg:
        out[season_avg_col] = series.groupby(
            [out["TEAM_ID"], out["SEASON_YEAR"], out["HOME"]]
        ).transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

        # If season-to-date is missing (first game), fall back to last-N home/away
        out[season_avg_col] = out[season_avg_col].fillna(out[last_n_homeaway_col])

    # Return to your preferred ordering
    out.sort_values(["GAME_DATE"], ascending=False, inplace=True)

    return out


def compute_season_std(df, param="PTS"):
    """
    Computes the season standard deviation for a given parameter (e.g., "PTS"),
    excluding the current row's game. The standard deviation is computed over
    all prior games (using expanding window after a shift).

    The DataFrame must have columns:
        TEAM_ID, SEASON_YEAR, HOME, GAME_DATE, and `param`.
    It sorts by (TEAM_ID, GAME_DATE) ascending so that the "previous" games appear first.

    After calling this function, df will have a new column:
        f"{param}_SEASON_BEFORE_STD"
    which contains the expanding standard deviation of `param` for each group,
    computed using only games before the current one.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the necessary columns.
    param : str, default "PTS"
        The column for which to compute the season standard deviation.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an additional column f"{param}_SEASON_BEFORE_STD".
    """
    season_std_col = f"{param}_SEASON_BEFORE_STD"

    # Sort by TEAM_ID and GAME_DATE ascending to ensure proper order.
    df = df.sort_values(["TEAM_ID", "GAME_DATE"], ascending=True).copy()

    # Compute expanding standard deviation after shifting by 1 to exclude the current game.
    df[season_std_col] = df.groupby(["TEAM_ID", "SEASON_YEAR", "HOME"])[
        param
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).std(ddof=0))

    # Optionally, sort the DataFrame back by GAME_DATE descending.
    df = df.sort_values(["GAME_DATE"], ascending=False).reset_index(drop=True)

    return df


def compute_rolling_weighted_stats(df, param="PTS", window = 10,group_by_season=True):
    """
    Computes weighted moving average for a given param, excluding the current row's game.

    Creates:
      - f"{param}_LAST_{window}_WMA_BEFORE"
      - f"{param}_LAST_HOME_AWAY_{window}_WMA_BEFORE"

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TEAM_ID, HOME, GAME_DATE, GAME_ID, SEASON_YEAR, and `param`.
    param : str
        The column for which to compute the weighted moving average.
    group_by_season : bool, default True
        If True, computes rolling stats within each SEASON_YEAR to prevent
        cross-season contamination. If False, allows rolling across seasons.

    Returns
    -------
    pd.DataFrame
        DataFrame with weighted moving average columns added.
    """

    last_n_wma_col = f"{param}_LAST_{window}_WMA_BEFORE"
    last_n_split_col = f"{param}_LAST_HOME_AWAY_{window}_WMA_BEFORE"

    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    sort_cols = ["TEAM_ID", "GAME_DATE", "GAME_ID"]
    if group_by_season:
        sort_cols = ["TEAM_ID", "SEASON_YEAR", "GAME_DATE", "GAME_ID"]

    out.sort_values(sort_cols, ascending=True, inplace=True)

    series = pd.to_numeric(out[param], errors="coerce")

    def weighted_moving_average(x: pd.Series) -> float:
        """
        Compute weighted moving average with consistent relative weighting.
        Uses exponential-like weights that scale properly for any window size.
        """
        n = len(x)
        if x.isna().all():
            return np.nan

        # Generate weights for the actual window size (1 to n)
        # This ensures consistent relative weighting regardless of window size
        w = np.arange(1, n + 1, dtype=float)

        mask = ~x.isna()
        return float((x[mask] * w[mask.to_numpy()]).sum() / w[mask.to_numpy()].sum())

    # Build groupby keys
    if group_by_season:
        group_keys_any = [out["TEAM_ID"], out["SEASON_YEAR"]]
        group_keys_homeaway = [out["TEAM_ID"], out["SEASON_YEAR"], out["HOME"]]
    else:
        group_keys_any = [out["TEAM_ID"]]
        group_keys_homeaway = [out["TEAM_ID"], out["HOME"]]

    # Any (team-level)
    out[last_n_wma_col] = series.groupby(group_keys_any).transform(
        lambda s: s.shift(1)
        .rolling(window, min_periods=1)
        .apply(weighted_moving_average, raw=False)
    )

    # Home / Away split
    out[last_n_split_col] = series.groupby(group_keys_homeaway).transform(
        lambda s: s.shift(1)
        .rolling(window, min_periods=1)
        .apply(weighted_moving_average, raw=False)
    )

    out.sort_values("GAME_DATE", ascending=False, inplace=True)

    return out


def get_team_param_value(row, team_id, param):
    """
    Returns the correct stat from the row for the given team,
    either param+'_HOME' or param+'_AWAY' depending on if the
    row's home team or away team is the given team_id.

    Example:
      If param='PTS' and the home team is team_id, returns row['PTS_HOME']
      If param='PTS' and the away team is team_id, returns row['PTS_AWAY']
    """
    if row["TEAM_ID_HOME"] == team_id:
        return row[f"{param}_HOME"]
    elif row["TEAM_ID_AWAY"] == team_id:
        return row[f"{param}_AWAY"]
    else:
        # This row does not involve the team, return NaN or 0 as you prefer.
        return float("nan")


def get_pre_game_averages(
    df, game_id, param="PTS", season_averages=False, n_past_games=5
):
    """
    For a given game_id, compute four averages of `param`:
      1) home_avg_any_games  : Last 5 games (home or away) for the home_team_id
      2) home_avg_home_games : Last 5 HOME games for the home_team_id
      3) away_avg_any_games  : Last 5 games (home or away) for the away_team_id
      4) away_avg_away_games : Last 5 AWAY games for the away_team_id

    *BUT* the DataFrame has columns like "PTS_HOME" and "PTS_AWAY"
     (rather than a single "PTS").

    All are strictly before the reference game's date (excludes that game itself).

    Args:
        df (pd.DataFrame): Must have columns:
            - "GAME_ID"
            - "GAME_DATE" (datetime recommended)
            - "TEAM_ID_HOME", "TEAM_ID_AWAY"
            - param+"_HOME", param+"_AWAY"
        game_id (str): The reference game from which we get the date and teams.
        param (str): The stat prefix, e.g. "PTS".
                     We'll look for param+"_HOME" and param+"_AWAY" in each row.
        n_past_games (int): How many previous games to consider (e.g. 5).

    Returns:
        (float, float, float, float):
          (home_avg_any_games, home_avg_home_games,
           away_avg_any_games, away_avg_away_games)
        or (NaN, NaN, NaN, NaN) if the reference game is not found.
    """

    # --- 1) Identify the reference game row and its date + teams ---
    game_mask = df["GAME_ID"] == game_id

    if not game_mask.any():
        print(f"No matching records for GAME_ID={game_id}")
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Assuming only one row matches game_id:
    row_game = df.loc[game_mask].iloc[0]
    home_team_id = row_game["TEAM_ID_HOME"]
    away_team_id = row_game["TEAM_ID_AWAY"]
    game_date = row_game["GAME_DATE"]
    season_type = row_game["SEASON_TYPE"]

    # --- 2) Sort the DataFrame by date descending (so .head() gets most recent) ---
    # For performance, you might do this sort once outside the function if calling repeatedly.

    # ==================================================================
    #                 Helper: compute last N games
    # ==================================================================
    def average_last_n_games(df_in, team_id, is_home_filter=None):
        """
        Returns the average of param for the last n_past_games rows in df_in
        that match (TEAM_ID_HOME==team_id or TEAM_ID_AWAY==team_id)
        and optionally filters by 'is_home_filter' if needed.

        If is_home_filter is True, team_id must be the home team in that row.
        If is_home_filter is False, team_id must be the away team in that row.
        If is_home_filter is None, any game featuring that team is allowed.
        """
        if is_home_filter is True:
            mask = (df_in["TEAM_ID_HOME"] == team_id) & (df_in["GAME_DATE"] < game_date)
        elif is_home_filter is False:
            mask = (df_in["TEAM_ID_AWAY"] == team_id) & (df_in["GAME_DATE"] < game_date)
        else:
            # is_home_filter is None => any game with team_id as home or away
            mask = (
                (df_in["TEAM_ID_HOME"] == team_id) | (df_in["TEAM_ID_AWAY"] == team_id)
            ) & (df_in["GAME_DATE"] < game_date)

        # Filter and sorting
        df_team = df_in[mask].sort_values(by="GAME_DATE", ascending=False)
        df_team = df_team.head(n_past_games)

        # For each row in df_team, pick param_HOME or param_AWAY
        # depending on whether the team_id is home or away in that row.
        selected_values = df_team.apply(
            lambda row: get_team_param_value(row, team_id, param), axis=1
        )

        return selected_values.mean()

    # --- 3) Compute each category's average using our helper ---

    # 3a) Home Team's last N ANY games (home or away)
    home_avg_any_games = average_last_n_games(df, home_team_id, is_home_filter=None)

    # 3b) Home Team's last N HOME games
    home_avg_home_games = average_last_n_games(df, home_team_id, is_home_filter=True)

    # 3c) Away Team's last N ANY games
    away_avg_any_games = average_last_n_games(df, away_team_id, is_home_filter=None)

    # 3d) Away Team's last N AWAY games
    away_avg_away_games = average_last_n_games(df, away_team_id, is_home_filter=False)

    # We'll store them in a dict
    results_dict = {
        f"{param.upper()}_HOME_AVG_LAST_{n_past_games}_ANY_GAMES": home_avg_any_games,
        f"{param.upper()}_HOME_AVG_LAST_{n_past_games}_HOME_GAMES": home_avg_home_games,
        f"{param.upper()}_AWAY_AVG_LAST_{n_past_games}_ANY_GAMES": away_avg_any_games,
        f"{param.upper()}_AWAY_AVG_LAST_{n_past_games}_AWAY_GAMES": away_avg_away_games,
    }

    # ----------------------------------------------------------------
    # 4) Optionally compute season_averages (two extra values)
    # ----------------------------------------------------------------
    if season_averages:
        # Home team's games in the same SEASON_TYPE, strictly before this game
        mask_home_season = (
            (df["TEAM_ID_HOME"] == home_team_id)
            & (df["SEASON_TYPE"] == season_type)
            & (df["GAME_DATE"] < game_date)
        )
        df_home_season = df[mask_home_season]

        # For each row, pick param_HOME or param_AWAY
        home_season_vals = df_home_season.apply(
            lambda row: get_team_param_value(row, home_team_id, param), axis=1
        )
        home_season_avg = home_season_vals.mean()

        # Away team's games in the same SEASON_TYPE, strictly before this game
        mask_away_season = (
            (df["TEAM_ID_AWAY"] == away_team_id)
            & (df["SEASON_TYPE"] == season_type)
            & (df["GAME_DATE"] < game_date)
        )
        df_away_season = df[mask_away_season]

        away_season_vals = df_away_season.apply(
            lambda row: get_team_param_value(row, away_team_id, param), axis=1
        )
        away_season_avg = away_season_vals.mean()

        # Add them to the results
        results_dict[f"{param.upper()}_HOME_SEASON_HOME_AVG"] = home_season_avg
        results_dict[f"{param.upper()}_AWAY_SEASON_AWAY_AVG"] = away_season_avg

    return results_dict


def precompute_cumulative_avg_stat(df_players, stat_col="PTS"):
    """
    Compute a cumulative average of the specified stat per (SEASON_ID, PLAYER_ID),
    considering only past games where they played (MIN > 0).

    - Excludes the current game's stat from the average.
    - If a player has not played before, their cumulative average is 0.
    - Only considers games where MIN > 0 as a valid appearance.
    - Groups by both SEASON_ID and PLAYER_ID.
    """
    # 1) Sort by season, player, then ascending game date
    df_players = df_players.sort_values(
        ["SEASON_ID", "PLAYER_ID", "GAME_DATE"], ascending=True
    ).copy()

    # 3) Identify valid games (where the player actually played)
    df_players["VALID_GAME"] = (
        (df_players["MIN"] > 0) & (df_players[stat_col].notna())
    ).astype(int)

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

    # 5) Compute cumulative sum of the stat (only from shifted “past” values)
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
    df_players[f"{stat_col}_CUM_AVG"] = np.where(
        df_players["GAME_COUNT"] > 0,
        df_players[f"CUMSUM_{stat_col}"] / df_players["GAME_COUNT"],
        0.0,
    )

    # 8) (Optional) Drop helper columns
    df_players.drop(
        columns=[shifted_stat_col, "VALID_GAME", "VALID_GAME_PREV"], inplace=True
    )

    return df_players


def get_injured_players_dict(df_injuries):
    """
    Build a dictionary: injured_dict[game_id][team_id] -> list of injured players for that game/team.
    """
    injured_dict = {}
    for game_id, df_g in df_injuries.groupby("GAME_ID"):
        team_map = {}
        for t_id, df_t in df_g.groupby("TEAM_ID"):
            team_map[t_id] = df_t["PLAYER_ID"].unique().tolist()
        injured_dict[game_id] = team_map
    return injured_dict


def _get_players_for_team_in_season(
    df_players, season_id, team_id, date_to_filter=None, filter_by_season_year=False
):
    """
    Example logic that returns rows from df_players belonging to (season_id, team_id),
    only for players who had not left by date_to_filter (based on last game).
    This is one approach; adapt to your logic as needed.
    """
    season_column = "SEASON_ID"
    if filter_by_season_year:
        season_column = "SEASON_YEAR"

    # Filter same season
    df_season = df_players[df_players[season_column] == season_id].copy()
    if df_season.empty:
        raise ValueError(f"No players found for {season_id}. Skipping...")
    # Only games BEFORE this date
    if date_to_filter is not None:
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
        (df_players[season_column] == season_id)
        & (df_players["TEAM_ID"] == team_id)
        & (df_players["PLAYER_ID"].isin(final_player_ids))
    ].copy()

    if date_to_filter is not None:
        df_result = df_result[df_result["GAME_DATE"] < date_to_filter]

    return df_result


def get_top_n_averages_with_names(
    df, date, stat_col="PTS", injured=False, lowest=False, n_players=3, min_minutes=15
):
    """
    Returns a list of tuples (Player Name, <CUM_AVG>) for the top 3 (or bottom 3) players
    by {stat_col}_CUM_AVG.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least: PLAYER_ID, GAME_DATE, PLAYER_NAME, {stat_col}_CUM_AVG
    date : datetime or str
        The target date (usually the current game date).
    stat_col : str
        The stat column (e.g., "PTS") for which we're looking up the cumulative average.
    injured : bool
        If False, we only consider players who played on `date`.
        If True, we instead consider the last game prior to `date`.
    lowest : bool
        If False (default), we return the highest three averages (descending).
        If True, we return the lowest three averages (ascending).
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

    # Merge the cumulative average minutes into the selected game rows.
    df_last = df_last.merge(df_cum_min, on="PLAYER_ID", how="left")

    cum_col = f"{stat_col}_CUM_AVG"

    # cREATE EXTRA VARIABLE to check if player meets the minimum threshold
    df_last["MEETS_MIN_THRESHOLD"] = (df_last["MIN_CUM_AVG"] >= min_minutes).astype(int)
    # Sort by the cumulative average column
    df_sorted = df_last.sort_values(
        by=["MEETS_MIN_THRESHOLD", cum_col], ascending=[False, lowest]
    )

    # Extract the top n (or bottom n) players
    chosen = df_sorted.head(n_players)

    top_or_bottom_n = list(zip(chosen["PLAYER_NAME"], chosen[cum_col]))

    return top_or_bottom_n


def attach_top3_stats(
    df_team, df_players, injured_dict, stat_cols=["PTS"], game_date_limit=None
):
    """
    Main function:
      - Precompute cumulative avg of `stat_cols` in df_players
      - Build an injured_dict
      - For each row in df_team, find top-3 average of `stat_cols` among non-injured and injured
        players who belong to that team on that date.
      - Return df_team with 12 new columns (including names).

    Parameters
    ----------
    df_team : pd.DataFrame
        DataFrame containing the team-level data (e.g., schedule, etc.)
    df_players : pd.DataFrame
        DataFrame with player-level boxscores (contains 'GAME_DATE', 'PLAYER_ID', etc.)
    df_injuries : pd.DataFrame
        DataFrame listing injuries per (GAME_ID, TEAM_ID, PLAYER_ID)
    stat_cols : str, default "PTS"
        The column (stat) for which we want to compute cumulative averages and find top-3.

    Returns
    -------
    pd.DataFrame
        Updated df_team with extra columns for top-3 players (names & average stat).
    """
    if isinstance(stat_cols, str):
        stat_cols = [stat_cols]

    for stat_col in stat_cols:
        # 1) Precompute cumulative averages for the chosen stat
        df_players[stat_col] = pd.to_numeric(df_players[stat_col], errors="coerce")
        df_players = precompute_cumulative_avg_stat(df_players, stat_col=stat_col)

    # 4) Iterate over each row in df_team
    for idx, row in tqdm(
        df_team.iterrows(), total=df_team.shape[0], desc="Adding players data"
    ):
        game_id = row["GAME_ID"]
        team_id = row["TEAM_ID"]
        game_date = row["GAME_DATE"]
        season_year = row["SEASON_YEAR"]

        if game_date_limit and game_date.strftime("%Y-%m-%d") != game_date_limit:
            continue

        else:
            print(f"Processing game {game_id} for team {team_id} on {game_date}...")

        df_active = _get_players_for_team_in_season(
            df_players=df_players,
            season_id=season_year,
            team_id=team_id,
            date_to_filter=game_date,
            filter_by_season_year=True,
        )

        if df_active.empty:
            print(f"No active players found for {team_id} on {game_date}. Skipping...")
            continue

        if not df_active.empty and (df_active["GAME_DATE"].max() >= game_date):
            raise AssertionError(
                "Leakage risk: df_active contains rows on or after game_date"
            )

        # Who is injured for this game/team?
        game_injured_map = injured_dict.get(game_id, {})
        injured_players = set(game_injured_map.get(team_id, []))

        # Separate non-injured and injured players
        df_non_inj = df_active[~df_active["PLAYER_ID"].isin(injured_players)]
        df_inj = df_active[df_active["PLAYER_ID"].isin(injured_players)]
        for stat_col in stat_cols:
            # Top-3 for each
            n_players_noninj = 8
            n_players_inj = 6
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

            # Pad to length 3, with Zeros for the numbers, and None for players
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
