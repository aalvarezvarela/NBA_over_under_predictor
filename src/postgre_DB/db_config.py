"""
Centralized database configuration module.
Reads database credentials from config.ini file.
"""

import configparser
from pathlib import Path

import psycopg


def get_predictions_db_name():
    """Get the predictions database name from config."""
    config = get_config()
    return config.get("Database", "DB_NAME_PREDICTIONS")


def connect_predictions_db():
    """Connect to the NBA predictions database."""
    credentials = get_db_credentials()
    db_name = get_predictions_db_name()
    return psycopg.connect(
        dbname=db_name,
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
    )


def get_config():
    """Load configuration from config.ini file."""
    # Get the project root directory (2 levels up from this file)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    config_path = project_root / "config.ini"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_db_credentials():
    """Get database credentials from config.ini."""
    config = get_config()

    credentials = {
        "user": config.get("Database", "DB_USER"),
        "password": config.get("Database", "DB_PASSWORD"),
        "host": config.get("Database", "DB_HOST"),
        "port": config.get("Database", "DB_PORT"),
    }
    return credentials


def get_games_db_name():
    """Get the games database name from config."""
    config = get_config()
    return config.get("Database", "DB_NAME_GAMES")


def get_players_db_name():
    """Get the players database name from config."""
    config = get_config()
    return config.get("Database", "DB_NAME_PLAYERS")


def get_odds_db_name():
    """Get the odds database name from config."""
    config = get_config()
    return config.get("Database", "DB_NAME_ODDS")


def connect_postgres_db():
    """Connect to PostgreSQL server (postgres database)."""
    credentials = get_db_credentials()
    return psycopg.connect(
        dbname="postgres",
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
        autocommit=True,
    )


def connect_games_db():
    """Connect to the NBA games database."""
    credentials = get_db_credentials()
    db_name = get_games_db_name()
    return psycopg.connect(
        dbname=db_name,
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
    )


def connect_players_db():
    """Connect to the NBA players database."""
    credentials = get_db_credentials()
    db_name = get_players_db_name()
    return psycopg.connect(
        dbname=db_name,
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
    )


def connect_odds_db():
    """Connect to the NBA odds database."""
    credentials = get_db_credentials()
    db_name = get_odds_db_name()
    return psycopg.connect(
        dbname=db_name,
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
    )
