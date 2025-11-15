"""
NBA Over/Under Predictor - Settings Module

This module handles application settings and configuration loading from config.ini.
It provides a centralized way to access all configuration parameters.
"""

import configparser
import os
from pathlib import Path

# Locate the config.ini file (assumes it's in the project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.ini"


class Settings:
    """
    Application settings loaded from config.ini file.

    This class reads configuration from the config.ini file and provides
    easy access to all application settings through properties.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize settings by loading from config.ini.

        Args:
            config_path: Optional custom path to config file.
                        If None, uses default location.
        """
        self.config = configparser.ConfigParser()

        if config_path:
            self.config.read(config_path)
        elif CONFIG_FILE.exists():
            self.config.read(CONFIG_FILE)
        else:
            raise FileNotFoundError(
                f"Config file not found at {CONFIG_FILE}. "
                "Please ensure config.ini exists in the project root."
            )

    # =========================================================================
    # PATHS
    # =========================================================================

    @property
    def output_path(self) -> str:
        """Directory for saving prediction outputs."""
        return self.config.get("Paths", "OUTPUT_PATH", fallback="./Predictions/")

    @property
    def regressor_model_path(self) -> str:
        """Path to the trained regressor model file."""
        return self.config.get(
            "Paths", "REGRESSOR_PATH", fallback="./model/production_regressor_xgb.pkl"
        )

    @property
    def classifier_model_path(self) -> str:
        """Path to the trained classifier model file."""
        return self.config.get(
            "Paths", "CLASSIFIER_PATH", fallback="./model/production_classifier_xgb.pkl"
        )

    @property
    def data_folder(self) -> str:
        """Directory containing NBA game data."""
        return self.config.get("Paths", "DATA_FOLDER", fallback="./data/")

    @property
    def report_path(self) -> str:
        """Directory for storing injury reports."""
        return self.config.get("Paths", "REPORT_PATH", fallback="./injury_reports/")

    @property
    def manual_data_entry_path(self) -> str:
        """Directory for manual odds data entry (optional)."""
        return self.config.get(
            "Paths", "MANUAL_DATA_ENTRY_PATH", fallback="./MANUAL ODDS ENTRY/"
        )

    # =========================================================================
    # INJURIES
    # =========================================================================

    @property
    def nba_injury_reports_url(self) -> str:
        """URL for fetching NBA injury reports."""
        return self.config.get(
            "Injuries",
            "NBA_INJURY_REPORTS_URL",
            fallback="https://official.nba.com/nba-injury-report-2024-25-season/",
        )

    # =========================================================================
    # ODDS
    # =========================================================================

    @property
    def odds_data_folder(self) -> str:
        """Directory for storing odds data."""
        return self.config.get("Odds", "ODDS_DATA_FOLDER", fallback="./odds_data/")

    @property
    def odds_api_key(self) -> str:
        """API key for accessing odds data."""
        return self.config.get("Odds", "ODDS_API_KEY", fallback="")

    @property
    def odds_base_url(self) -> str:
        """Base URL for the odds API."""
        return self.config.get(
            "Odds",
            "BASE_URL",
            fallback="https://therundown-therundown-v1.p.rapidapi.com",
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Convert a relative path from config to an absolute path.

        Args:
            relative_path: Path relative to project root

        Returns:
            Absolute Path object
        """
        if os.path.isabs(relative_path):
            return Path(relative_path)
        return PROJECT_ROOT / relative_path

    def ensure_directories_exist(self):
        """
        Create all necessary directories if they don't exist.
        """
        directories = [
            self.output_path,
            self.data_folder,
            self.report_path,
            self.odds_data_folder,
        ]

        for directory in directories:
            abs_path = self.get_absolute_path(directory)
            abs_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        """String representation of settings."""
        return (
            f"Settings(\n"
            f"  output_path='{self.output_path}',\n"
            f"  data_folder='{self.data_folder}',\n"
            f"  regressor_model='{self.regressor_model_path}',\n"
            f"  classifier_model='{self.classifier_model_path}'\n"
            f")"
        )


# Create a default settings instance for easy importing
settings = Settings()


# For backward compatibility and convenience
def get_settings(config_path: str = None) -> Settings:
    """
    Get a Settings instance.

    Args:
        config_path: Optional custom path to config file

    Returns:
        Settings instance
    """
    if config_path:
        return Settings(config_path)
    return settings
