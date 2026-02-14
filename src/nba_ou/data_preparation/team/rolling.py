import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

from nba_ou.data_preparation.statistics.statistics import (
    compute_rolling_stats,
    compute_rolling_weighted_stats,
    compute_season_std,
)

# Module-level constants for rolling statistics computation
COLS_TO_AVERAGE = [
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
    "PF",
]

COLS_TO_AVERAGE_ODDS = [
    "TOTAL_OVER_UNDER_LINE",
    "DIFF_FROM_LINE",
    "TOTAL_POINTS",
    "MONEYLINE",
    "SPREAD",
    "IS_OVER_LINE",
]

COLS_FOR_WEIGHTED_STATS = [
    "PTS",
    "TOTAL_POINTS",
    "TOTAL_OVER_UNDER_LINE",
    "DIFF_FROM_LINE",
]

COLS_FOR_SEASON_STD = [
    "PTS",
    "TOTAL_POINTS",
    "TOTAL_OVER_UNDER_LINE",
    "DIFF_FROM_LINE",
    "IS_OVER_LINE",
]


def compute_trend_slope(df, parameter="PTS", window=10, shift_current_game=True):
    """
    Computes the slope of a linear regression line over the last `window` games
    to determine whether a team's performance is increasing, decreasing, or stable.

    Works on a dataframe with one row per team per game (2 rows per match).

    Args:
        df (pd.DataFrame): Must contain columns "TEAM_ID", "SEASON_YEAR", "GAME_DATE", "HOME", and `parameter`.
        parameter (str): The statistic to analyze (e.g., "PTS").
        window (int): Number of last games to consider.
        shift_current_game (bool): Whether to exclude the current game from the trend calculation.

    Returns:
        pd.DataFrame: A modified DataFrame with new columns:
            - f"{parameter}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE" (all games)
            - f"{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_GAMES_BEFORE" (home trend if playing home, away trend if playing away)
    """

    def calculate_slope(series):
        """Applies linear regression to compute the trend slope."""
        # Remove NaN and None values
        clean_series = [x for x in series if x is not None and not np.isnan(x)]

        if len(clean_series) < 2:
            return 0  # Not enough data for a trend

        X = np.arange(1, len(clean_series) + 1)  # Time index [1, 2, ..., N]
        Y = np.array(clean_series)  # Convert to array for linregress

        slope, _, _, _, _ = linregress(X, Y)
        return slope

    # Sort by team, season, and date
    df = df.sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True)

    # 1. Overall trend (all games for each team)
    trend_col = f"{parameter}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"
    df[trend_col] = df.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter].transform(
        lambda s: (s.shift(1) if shift_current_game else s)
        .rolling(window, min_periods=2)
        .apply(calculate_slope, raw=True)
    )

    # 2. Location-specific trend (home trend when playing home, away trend when playing away)
    trend_col_location = f"{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_GAMES_BEFORE"

    # Compute home games trend for home teams
    home_mask = df["HOME"] == 1
    df_temp_home = df[home_mask].copy()
    df_temp_home = df_temp_home.sort_values(
        ["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True
    )

    df.loc[home_mask, trend_col_location] = (
        df_temp_home.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter]
        .transform(
            lambda s: (s.shift(1) if shift_current_game else s)
            .rolling(window, min_periods=2)
            .apply(calculate_slope, raw=True)
        )
        .values
    )

    # Compute away games trend for away teams
    away_mask = df["HOME"] == 0
    df_temp_away = df[away_mask].copy()
    df_temp_away = df_temp_away.sort_values(
        ["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True
    )

    df.loc[away_mask, trend_col_location] = (
        df_temp_away.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter]
        .transform(
            lambda s: (s.shift(1) if shift_current_game else s)
            .rolling(window, min_periods=2)
            .apply(calculate_slope, raw=True)
        )
        .values
    )

    return df


def compute_all_rolling_statistics(df):
    """
    Compute rolling statistics, weighted averages, and seasonal standard deviations,
    dynamically including new DIFF_FROM_* columns derived from extra total lines.

    Only adds new DIFF_FROM_* columns (not IS_OVER_*).
    """
    # 1) Dynamically discover new diff columns (exclude legacy DIFF_FROM_LINE)
    new_diff_cols = [
        c for c in df.columns if c.startswith("DIFF_FROM_") and c != "DIFF_FROM_LINE"
    ]

    # Optional: ensure we only include diffs that correspond to totals lines you created
    # (keeps things tight if you have other DIFF_FROM_* features in the future)
    total_line_cols = set(
        [
            c
            for c in df.columns
            if c == "TOTAL_OVER_UNDER_LINE" or c.startswith("TOTAL_LINE_")
        ]
    )
    allowed_suffixes = set()
    for tl in total_line_cols:
        allowed_suffixes.add(
            tl.replace("TOTAL_", "")
        )  # OVER_UNDER_LINE or LINE_betmgm, etc.
    new_diff_cols = [
        c for c in new_diff_cols if c.replace("DIFF_FROM_", "") in allowed_suffixes
    ]

    # 2) Build local versions of your lists (do not mutate module-level constants)
    cols_to_average_odds = COLS_TO_AVERAGE_ODDS + new_diff_cols

    cols_for_weighted_stats = COLS_FOR_WEIGHTED_STATS

    cols_for_season_std = COLS_FOR_SEASON_STD + new_diff_cols
    cols_for_season_std = list(dict.fromkeys(cols_for_season_std))

    # 3) Rolling stats loop
    for col in tqdm(
        COLS_TO_AVERAGE + cols_to_average_odds, desc="Computing rolling stats"
    ):
        df = compute_rolling_stats(
            df, col, window=5, add_extra_season_avg=True, group_by_season=False
        )
        df = compute_rolling_stats(
            df, col, window=10, add_extra_season_avg=False, group_by_season=False
        )

        if col in cols_for_weighted_stats:
            df = compute_rolling_weighted_stats(
                df, col, window=10, group_by_season=False
            )
            df = compute_rolling_weighted_stats(
                df, col, window=5, group_by_season=False
            )

        # Extra short windows for all DIFF columns (legacy + new ones)
        if col == "DIFF_FROM_LINE" or col in new_diff_cols:
            df = compute_rolling_stats(df, col, window=1, add_extra_season_avg=False)
            df = compute_rolling_stats(df, col, window=2, add_extra_season_avg=False)
            df = compute_rolling_stats(df, col, window=3, add_extra_season_avg=False)

    # 4) Seasonal std loop
    for param in cols_for_season_std:
        df = compute_season_std(df, param=param)

    # Compute trend slopes for teams
    df = compute_trend_slope(df, parameter="PTS", window=10, shift_current_game=True)
    df = compute_trend_slope(df, parameter="PTS", window=5, shift_current_game=True)

        # Diff-from-line columns (post-game): exclude current game
    diff_cols = [c for c in df.columns if c.startswith("DIFF_FROM_")]
    # Compute trends for all diffs (these depend on TOTAL_POINTS)
    for col in diff_cols:
        df = compute_trend_slope(df, parameter=col, window=5, shift_current_game=True)
    
    # Total line columns (pre-game known): include current game
    total_line_cols = ["TOTAL_OVER_UNDER_LINE"] + [c for c in df.columns if c.startswith("TOTAL_LINE_")]
    # Compute trends for all total lines
    for col in total_line_cols:
        if col in df.columns:
            df = compute_trend_slope(df, parameter=col, window=5, shift_current_game=False)

    return df

