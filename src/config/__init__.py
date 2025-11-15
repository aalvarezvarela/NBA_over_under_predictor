"""
NBA Over/Under Predictor - Configuration Package

This package contains all configuration, constants, and settings for the NBA prediction system.
"""

from .constants import (  # Team mappings; Season types; Game configuration; Statistical windows; External APIs; Backward compatibility
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_TOP_PLAYERS,
    MIN_MINUTES_THRESHOLD,
    NBA_INJURY_REPORTS_URL,
    NBA_SPORT_ID,
    OVERTIME_THRESHOLD_MINUTES,
    REGULATION_GAME_MINUTES,
    SEASON_TYPE_MAP,
    SEASON_TYPE_MAPPING,
    TEAM_CONVERSION_DICT,
    TEAM_ID_MAP,
    TEAM_NAME_EQUIVALENT_DICT,
    TEAM_NAME_STANDARDIZATION,
    WEIGHTED_MA_WEIGHTS,
    WEIGHTED_ROLLING_WINDOW,
)

__all__ = [
    # Team mappings
    "TEAM_ID_MAP",
    "TEAM_NAME_STANDARDIZATION",
    # Season types
    "SEASON_TYPE_MAP",
    # Game configuration
    "REGULATION_GAME_MINUTES",
    "OVERTIME_THRESHOLD_MINUTES",
    # Statistical windows
    "DEFAULT_ROLLING_WINDOW",
    "WEIGHTED_ROLLING_WINDOW",
    "WEIGHTED_MA_WEIGHTS",
    "MIN_MINUTES_THRESHOLD",
    "DEFAULT_TOP_PLAYERS",
    # External APIs
    "NBA_INJURY_REPORTS_URL",
    "NBA_SPORT_ID",
    # Backward compatibility
    "TEAM_CONVERSION_DICT",
    "TEAM_NAME_EQUIVALENT_DICT",
    "SEASON_TYPE_MAPPING",
]
