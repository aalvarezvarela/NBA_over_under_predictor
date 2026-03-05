import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

from nba_ou.config.odds_columns import moneyline_col, spread_col, total_line_col
from nba_ou.data_preparation.statistics.statistics import (
    compute_rolling_stats,
    compute_rolling_weighted_stats,
    compute_season_std,
)

MAIN_TOTAL_LINE_COL = total_line_col()
MAIN_SPREAD_COL = spread_col()
MAIN_MONEYLINE_COL = moneyline_col()

# Module-level constants for rolling statistics computation
COLS_TO_AVERAGE = [
    "PTS",
    "TOTAL_POINTS",
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
    MAIN_TOTAL_LINE_COL,
    "TOTAL_POINTS",
    MAIN_MONEYLINE_COL,
    MAIN_SPREAD_COL,
]

COLS_FOR_WEIGHTED_STATS = [
    "PTS",
    "TOTAL_POINTS",
    MAIN_TOTAL_LINE_COL,
]

COLS_FOR_SEASON_STD = [
    "PTS",
    "TOTAL_POINTS",
    MAIN_TOTAL_LINE_COL,
]

COLS_FOR_SHORT_WINDOWS = [
    "PTS",
    "DIFF_FROM_LINE_bet365",
    MAIN_TOTAL_LINE_COL,
]


def compute_trend_slope(
    df,
    parameter="PTS",
    window=10,
    shift_current_game=True,
    add_relative_column: bool = True,
    include_home_away_relative: bool = True,
    relative_to_window: int | None = None,
):
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
            - f"{parameter}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE" (strict all-games trend)
            - Relative column (if add_relative_column=True):
                - home/away minus strict (legacy name reused):
                  f"{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_GAMES_BEFORE"
                - OR strict-window diff:
                  f"{parameter}_TREND_SLOPE_LAST_{relative_to_window}_MINUS_LAST_{window}_GAMES_BEFORE"
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

    if add_relative_column:
        if include_home_away_relative:
            # 2A. Relative trend using home/away context:
            # (home/away trend) - (strict all-games trend)
            trend_col_location = (
                f"{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_GAMES_BEFORE"
            )
            trend_col_location_raw = (
                f"__{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_RAW"
            )

            home_mask = df["HOME"] == 1
            df_temp_home = df[home_mask].copy().sort_values(
                ["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True
            )
            df.loc[home_mask, trend_col_location_raw] = (
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

            away_mask = df["HOME"] == 0
            df_temp_away = df[away_mask].copy().sort_values(
                ["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True
            )
            df.loc[away_mask, trend_col_location_raw] = (
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

            df[trend_col_location] = df[trend_col_location_raw] - df[trend_col]
            df.drop(columns=[trend_col_location_raw], inplace=True)
        elif relative_to_window is not None:
            # 2B. Relative strict trend across windows (e.g., last_5 - last_10)
            relative_col = (
                f"{parameter}_TREND_SLOPE_LAST_{relative_to_window}_MINUS_LAST_{window}_GAMES_BEFORE"
            )
            if relative_to_window <= 0:
                raise ValueError("relative_to_window must be > 0 when provided.")

            ref_col = (
                f"{parameter}_TREND_SLOPE_LAST_{relative_to_window}_GAMES_BEFORE"
            )
            if ref_col in df.columns:
                reference_trend = df[ref_col]
            else:
                reference_trend = df.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter].transform(
                    lambda s: (
                        (s.shift(1) if shift_current_game else s)
                        .rolling(relative_to_window, min_periods=2)
                        .apply(calculate_slope, raw=True)
                    )
                )

            df[relative_col] = reference_trend - df[trend_col]

    return df


def compute_all_rolling_statistics(df, exclude_yahoo=False):
    """
    Compute rolling statistics, weighted averages, and seasonal standard deviations,
    dynamically including new DIFF_FROM_* columns, TOTAL_LINE_* columns, and all odds data.

    Args:
        df (pd.DataFrame): DataFrame with game statistics
        exclude_yahoo (bool): If True, exclude Yahoo-specific betting columns (pct_bets, pct_money)
                             from rolling statistics. Default is False (include Yahoo columns).

    Includes all total line columns (TOTAL_LINE_*) and their
    corresponding DIFF_FROM_* columns in rolling stats, weighted stats, and season std.
    Also includes odds percentages, prices, and other betting data.
    """
    original_columns = set(df.columns)

    # 1) Dynamically discover new diff columns
    new_diff_cols = [
        c for c in df.columns if c.startswith("DIFF_FROM_")
    ]

    # 2) Dynamically discover all TOTAL_LINE_* columns
    new_total_line_cols = [c for c in df.columns if c.startswith("TOTAL_LINE_")]

    # 3) Dynamically discover Yahoo betting columns (percentage of bets/money)
    # Note: spread/ml yahoo columns are now team-specific (without _home/_away suffix after merge)
    yahoo_cols = []
    if not exclude_yahoo:
        yahoo_patterns = ["_pct_bets", "_pct_money"]
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
    # Note: spread/ml prices are now team-specific (without _home/_away suffix after merge)
    price_cols = [
        c
        for c in df.columns
        if "_price" in c
        and (c.startswith("total_") or c.startswith("spread_") or c.startswith("ml_"))
    ]

    # 6) Dynamically discover IS_OVER_* columns (excluding legacy IS_OVER_LINE)


    # 7) Optional: ensure we only include diffs that correspond to totals lines
    # (keeps things tight if you have other DIFF_FROM_* features in the future)
    total_line_cols = {c for c in df.columns if c.startswith("TOTAL_LINE_")}
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
    )
    cols_for_season_std = list(dict.fromkeys(cols_for_season_std))

    # 9) Rolling stats loop
    for col in tqdm(
        COLS_TO_AVERAGE + cols_to_average_odds, desc="Computing rolling stats"
    ):
        df = compute_rolling_stats(
            df, col, window=5, add_extra_season_avg=True, group_by_season=False
        )
        
        if col in COLS_FOR_SHORT_WINDOWS + new_total_line_cols + new_diff_cols + consensus_pct_cols:
            df = compute_rolling_stats(
                df,
                col,
                window=10,
                add_extra_season_avg=False,
                group_by_season=False,
                include_home_away_relative=False,
                relative_to_window=5,
            )

        if col in cols_for_weighted_stats:
            df = compute_rolling_weighted_stats(
                df, col, window=5, group_by_season=False
            )

        # Extra short windows for all DIFF columns (legacy + new ones)
        if col in COLS_FOR_SHORT_WINDOWS + new_diff_cols:
            df = compute_rolling_stats(
                df,
                col,
                window=1,
                add_extra_season_avg=False,
                add_relative_column=False,
            )
            df = compute_rolling_stats(
                df,
                col,
                window=2,
                add_extra_season_avg=False,
                add_relative_column=False,
            )
            df = compute_rolling_stats(
                df,
                col,
                window=3,
                add_extra_season_avg=False,
                add_relative_column=False,
            )

    # 10) Seasonal std loop
    for param in tqdm(cols_for_season_std, desc="Computing seasonal std"):
        df = compute_season_std(df, param=param)

    # 11) Compute trend slopes for teams
    print("Computing team performance trends...")
    df = compute_trend_slope(df, parameter="PTS", window=5, shift_current_game=True)
    df = compute_trend_slope(
        df,
        parameter="PTS",
        window=10,
        shift_current_game=True,
        include_home_away_relative=False,
        relative_to_window=5,
    )

    # 12) Diff-from-line columns (post-game): exclude current game
    diff_cols = [c for c in df.columns if c.startswith("DIFF_FROM_")]
    for col in tqdm(diff_cols, desc="Computing diff-from-line trends"):
        df = compute_trend_slope(df, parameter=col, window=5, shift_current_game=True)

    # 13) Total line columns (pre-game known): trends shift based on column
    total_line_trend_cols = [c for c in df.columns if c.startswith("TOTAL_LINE_")]
    if MAIN_TOTAL_LINE_COL in total_line_trend_cols:
        total_line_trend_cols = [MAIN_TOTAL_LINE_COL] + [
            c for c in total_line_trend_cols if c != MAIN_TOTAL_LINE_COL
        ]
    for col in tqdm(total_line_trend_cols, desc="Computing total line trends"):
        if col in df.columns:
            df = compute_trend_slope(
                df, parameter=col, window=5, shift_current_game=True
            )

    # 14) Consensus percentage trends (pre-game known): shift current game
    for col in tqdm(consensus_pct_cols, desc="Computing consensus % trends"):
        if col in df.columns:
            df = compute_trend_slope(
                df, parameter=col, window=5, shift_current_game=True
            )

    # 15) Yahoo percentage trends (if included)
    if not exclude_yahoo:
        for col in tqdm(yahoo_cols, desc="Computing Yahoo % trends"):
            if col in df.columns:
                df = compute_trend_slope(
                    df, parameter=col, window=5, shift_current_game=True
                )

    # 16) Enforce naming convention: all newly computed rolling/stat columns must contain _BEFORE
    # Only apply to columns created inside this function (keep source columns untouched).
    new_columns = set(df.columns) - original_columns
    rename_map = {}
    for col in new_columns:
        is_computed_stat = (
            ("_LAST_" in col and ("_MATCHES" in col or "_WMA_" in col))
            or ("_TREND_SLOPE_" in col)
            or ("_SEASON_" in col and (col.endswith("_AVG") or col.endswith("_STD")))
        )
        if is_computed_stat and "_BEFORE" not in col:
            rename_map[col] = f"{col}_BEFORE"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df
