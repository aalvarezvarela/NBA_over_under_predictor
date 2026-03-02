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
        lambda s: (
            (s.shift(1) if shift_current_game else s)
            .rolling(window, min_periods=2)
            .apply(calculate_slope, raw=True)
        )
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
            lambda s: (
                (s.shift(1) if shift_current_game else s)
                .rolling(window, min_periods=2)
                .apply(calculate_slope, raw=True)
            )
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
            lambda s: (
                (s.shift(1) if shift_current_game else s)
                .rolling(window, min_periods=2)
                .apply(calculate_slope, raw=True)
            )
        )
        .values
    )

    return df


def compute_all_rolling_statistics(df, exclude_yahoo=False):
    """
    Compute rolling statistics, weighted averages, and seasonal standard deviations,
    dynamically including new DIFF_FROM_* columns, TOTAL_LINE_* columns, and all odds data.

    Args:
        df (pd.DataFrame): DataFrame with game statistics
        exclude_yahoo (bool): If True, exclude Yahoo-specific betting columns (pct_bets, pct_money)
                             from rolling statistics. Default is False (include Yahoo columns).

    Includes all total line columns (TOTAL_OVER_UNDER_LINE + TOTAL_LINE_*) and their
    corresponding DIFF_FROM_* columns in rolling stats, weighted stats, and season std.
    Also includes odds percentages, prices, and other betting data.
    """
    # 1) Dynamically discover new diff columns (exclude legacy DIFF_FROM_LINE)
    new_diff_cols = [
        c for c in df.columns if c.startswith("DIFF_FROM_") and c != "DIFF_FROM_LINE"
    ]

    # 2) Dynamically discover all TOTAL_LINE_* columns
    new_total_line_cols = [c for c in df.columns if c.startswith("TOTAL_LINE_")]

    # 3) Dynamically discover Yahoo betting columns (percentage of bets/money)
    yahoo_cols = []
    if not exclude_yahoo:
        yahoo_patterns = ["_pct_bets_", "_pct_money_"]
        yahoo_cols = [
            c for c in df.columns if any(pattern in c for pattern in yahoo_patterns)
        ]

    # 4) Dynamically discover consensus percentage columns
    consensus_pct_cols = [
        c
        for c in df.columns
        if (
            c.startswith("total_consensus_pct_")
            or c.startswith("spread_consensus_pct_")
            or c.startswith("moneyline_consensus_pct_")
        )
    ]

    # 5) Dynamically discover price columns (odds prices for totals, spreads, moneylines)
    price_cols = [
        c
        for c in df.columns
        if (
            "_price_" in c
            and (
                c.startswith("total_") or c.startswith("spread_") or c.startswith("ml_")
            )
        )
    ]

    # 6) Dynamically discover IS_OVER_* columns (excluding legacy IS_OVER_LINE)
    is_over_cols = [
        c for c in df.columns if c.startswith("IS_OVER_") and c != "IS_OVER_LINE"
    ]

    # 7) Optional: ensure we only include diffs that correspond to totals lines
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

    # 8) Build local versions of lists (do not mutate module-level constants)
    cols_to_average_odds = (
        COLS_TO_AVERAGE_ODDS
        + new_diff_cols
        + new_total_line_cols
        + yahoo_cols
        + consensus_pct_cols
        + price_cols
        + is_over_cols
    )

    # Weighted stats: include total lines, prices, and consensus percentages
    cols_for_weighted_stats = (
        COLS_FOR_WEIGHTED_STATS + new_total_line_cols + consensus_pct_cols
    )

    # Season std: include diffs, total lines, yahoo, consensus, and is_over columns
    cols_for_season_std = (
        COLS_FOR_SEASON_STD
        + new_diff_cols
        + new_total_line_cols
        + yahoo_cols
        + consensus_pct_cols
        + is_over_cols
    )
    cols_for_season_std = list(dict.fromkeys(cols_for_season_std))

    # 9) Rolling stats loop
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

    # 10) Seasonal std loop
    for param in cols_for_season_std:
        df = compute_season_std(df, param=param)

    # 11) Compute trend slopes for teams
    df = compute_trend_slope(df, parameter="PTS", window=10, shift_current_game=True)
    df = compute_trend_slope(df, parameter="PTS", window=5, shift_current_game=True)

    # 12) Diff-from-line columns (post-game): exclude current game
    diff_cols = [c for c in df.columns if c.startswith("DIFF_FROM_")]
    for col in diff_cols:
        df = compute_trend_slope(df, parameter=col, window=5, shift_current_game=True)

    # 13) Total line columns (pre-game known): trends shift based on column
    total_line_trend_cols = ["TOTAL_OVER_UNDER_LINE"] + [
        c for c in df.columns if c.startswith("TOTAL_LINE_")
    ]
    for col in total_line_trend_cols:
        if col in df.columns:
            if col == "TOTAL_OVER_UNDER_LINE":
                df = compute_trend_slope(
                    df, parameter=col, window=6, shift_current_game=False
                )
            else:
                df = compute_trend_slope(
                    df, parameter=col, window=5, shift_current_game=True
                )

    # 14) Consensus percentage trends (pre-game known): shift current game
    for col in consensus_pct_cols:
        if col in df.columns:
            df = compute_trend_slope(
                df, parameter=col, window=5, shift_current_game=True
            )

    # 15) Yahoo percentage trends (if included)
    if not exclude_yahoo:
        for col in yahoo_cols:
            if col in df.columns:
                df = compute_trend_slope(
                    df, parameter=col, window=5, shift_current_game=True
                )

    # 16) Price columns trends (pre-game known): shift current game
    for col in price_cols:
        if col in df.columns:
            df = compute_trend_slope(
                df, parameter=col, window=5, shift_current_game=True
            )

    return df
