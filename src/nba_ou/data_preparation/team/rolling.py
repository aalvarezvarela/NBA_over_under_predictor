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
]

COLS_FOR_WEIGHTED_STATS = ["PTS", "TOTAL_POINTS", "TOTAL_OVER_UNDER_LINE"]

COLS_FOR_SEASON_STD = ["PTS", "TOTAL_POINTS", "TOTAL_OVER_UNDER_LINE", "DIFF_FROM_LINE"]

def compute_all_rolling_statistics(df):
    """
    Compute rolling statistics, weighted averages, and seasonal standard deviations.

    This function:
    - Computes 5-game rolling averages for all specified columns
    - Computes 5 and 10-game weighted rolling stats for PTS, TOTAL_POINTS, and TOTAL_OVER_UNDER_LINE
    - Computes seasonal standard deviations for key parameters

    Args:
        df (pd.DataFrame): Team statistics DataFrame

    Returns:
        pd.DataFrame: DataFrame with added rolling statistics columns
    """
    # Compute rolling statistics for all columns
    for col in tqdm(
        COLS_TO_AVERAGE + COLS_TO_AVERAGE_ODDS, desc="Computing rolling stats"
    ):
        df = compute_rolling_stats(df, col, window=5, add_extra_season_avg=True)
        if col in COLS_FOR_WEIGHTED_STATS:
            df = compute_rolling_weighted_stats(
                df, col, window=10, group_by_season=False
            )
            df = compute_rolling_weighted_stats(
                df, col, window=5, group_by_season=False
            )

    # Compute seasonal standard deviations
    for param in COLS_FOR_SEASON_STD:
        df = compute_season_std(df, param=param)

    return df
