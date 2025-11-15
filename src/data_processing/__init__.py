"""
NBA Over/Under Predictor - Data Processing Package

This package contains modules for processing NBA game data, computing statistics,
and handling injury reports.
"""

from .injury_processing import (
    convert_name,
    get_player_id,
    process_injury_data,
    retrieve_injury_report_as_df,
)
from .process_data_with_injuries import (
    create_df_to_predict,
    get_last_two_nba_seasons,
    get_schedule_games,
    load_all_nba_data,
)
from .statistics import (
    attach_top3_stats,
    classify_season_type,
    compute_rolling_stats,
    compute_rolling_weighted_stats,
    compute_season_std,
    get_pre_game_averages,
    precompute_cumulative_avg_stat,
)

__all__ = [
    # Injury processing
    "process_injury_data",
    "retrieve_injury_report_as_df",
    "get_player_id",
    "convert_name",
    # Statistics
    "classify_season_type",
    "compute_rolling_stats",
    "compute_rolling_weighted_stats",
    "compute_season_std",
    "get_pre_game_averages",
    "precompute_cumulative_avg_stat",
    "attach_top3_stats",
    # Main processing
    "get_schedule_games",
    "get_last_two_nba_seasons",
    "load_all_nba_data",
    "create_df_to_predict",
]
