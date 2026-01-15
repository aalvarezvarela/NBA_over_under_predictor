"""
NBA Over/Under Predictor - Games Database Management Package

This package contains utilities for fetching and updating NBA game data from the NBA API.
"""

from utils.general_utils import get_nba_season_nullable

from .update_database import update_database
from .update_database_utils import (
    classify_season_type,
    fetch_box_score_data,
    fetch_nba_data,
    merge_stats,
    reset_nba_http_session,
)

__all__ = [
    "update_database",
    "get_nba_season_nullable",
    "reset_nba_http_session",
    "classify_season_type",
    "fetch_box_score_data",
    "merge_stats",
    "fetch_nba_data",
]
