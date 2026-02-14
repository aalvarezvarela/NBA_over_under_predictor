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
CONFIG_FILE = PROJECT_ROOT / "nba_ou/config.ini"
SECRETS_FILE = PROJECT_ROOT / "nba_ou/config.secrets.ini"


class Settings:
    def __init__(self, config_path: str | None = None, secrets_path: str | None = None):
        self.config = configparser.ConfigParser()

        main_path = Path(config_path) if config_path else CONFIG_FILE
        if not main_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {main_path}. "
                "Please ensure config.ini exists in the project root."
            )

        # 1) Load committed config
        self.config.read(main_path)

        # 2) Overlay optional secrets config (local)
        secrets_path = Path(secrets_path) if secrets_path else SECRETS_FILE
        if secrets_path.exists():
            self.config.read(secrets_path)

    # -------------------------------------------------------------------------
    # Secret resolver
    # -------------------------------------------------------------------------
    def _get_secret(
        self,
        section: str,
        key: str,
        env_var: str,
        *,
        required: bool = True,
        fallback: str | None = None,
    ) -> str:
        """
        Resolve secrets with priority:
          1) Environment variable
          2) config.ini/config.secrets.ini (overlay)
          3) fallback (if provided)
        """
        v = os.getenv(env_var)
        if v is not None and v != "":
            return v

        if self.config.has_option(section, key):
            v = self.config.get(section, key).strip()
            if v != "":
                return v

        if fallback is not None:
            return fallback

        if required:
            raise ValueError(
                f"Missing required secret [{section}] {key}. "
                f"Set env var {env_var} or define it in config.secrets.ini."
            )

        return ""

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
        )

    # =========================================================================
    # REFS
    # =========================================================================

    @property
    def nba_official_assignments_url(self) -> str:
        """URL for fetching NBA referee assignments."""
        return self.config.get(
            "Refs",
            "NBA_OFFICIAL_ASSIGNMENTS_URL",
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
        # Env var in CI, secrets file locally
        return self._get_secret("Odds", "ODDS_API_KEY", env_var="ODDS_API_KEY")

    @property
    def odds_base_url(self) -> str:
        """Base URL for the odds API."""
        return self.config.get(
            "Odds",
            "BASE_URL",
            fallback="https://therundown-therundown-v1.p.rapidapi.com",
        )

    @property
    def odds_save_pickle(self) -> bool:
        """Whether to save raw odds pickles."""
        return self.config.getboolean("Odds", "SAVE_ODDS_PICKLE", fallback=False)

    @property
    def odds_pickle_path(self) -> str | None:
        """Path to store raw odds pickles (optional)."""
        return self.config.get("Odds", "ODDS_PICKLE_PATH", fallback=None)
    @property
    def s3_aws_region(self) -> str:
        return self.config.get("S3", "AWS_REGION", fallback="eu-west-1")

    @property
    def s3_bucket(self) -> str:
        return self.config.get("S3", "BUCKET")

    @property
    def s3_regressor_s3_key(self) -> str:
        return self.config.get("S3", "REGRESSOR_S3_KEY")

    @property
    def s3_regressor_meta_s3_key(self) -> str | None:
        return self.config.get("S3", "REGRESSOR_META_S3_KEY", fallback=None)

    @property
    def s3_aws_profile(self) -> str | None:
        """
        Local-only convenience. In CI (GitHub Actions with OIDC),
        you typically do NOT want/need a profile.
        """
        return self.config.get("S3", "AWS_PROFILE", fallback=None)

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
SETTINGS = Settings()


# For backward compatibility and convenience
def get_settings(config_path: str | None = None) -> Settings:
    if config_path:
        return Settings(config_path)
    return SETTINGS
