"""
Centralized database configuration module.
Reads database credentials from config.ini file.
"""

import configparser
from pathlib import Path

import psycopg
from psycopg import sql


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
    return {
        "user": config.get("Database", "DB_USER"),
        "password": config.get("Database", "DB_PASSWORD"),
        "host": config.get("Database", "DB_HOST"),
        "port": config.get("Database", "DB_PORT"),
        "dbname": config.get("Database", "DB_NAME").strip().strip('"').strip("'"),
    }


def get_schema_name_games():
    config = get_config()
    return config.get("Database", "SCHEMA_NAME_GAMES")


def get_schema_name_players():
    config = get_config()
    return config.get("Database", "SCHEMA_NAME_PLAYERS")


def get_schema_name_odds():
    config = get_config()
    return config.get("Database", "SCHEMA_NAME_ODDS")


def get_schema_name_predictions():
    config = get_config()
    return config.get("Database", "SCHEMA_NAME_PREDICTIONS")


def connect_postgres_db():
    """Connect to PostgreSQL server (postgres database) for admin tasks."""
    credentials = get_db_credentials()
    return psycopg.connect(
        dbname="postgres",
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
        autocommit=True,
    )


def connect_nba_db():
    """Connect to the single NBA database."""
    credentials = get_db_credentials()
    return psycopg.connect(
        dbname=credentials["dbname"],
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
    )


def connect_schema_db(schema: str):
    """
    Connect to the single NBA database and set search_path to the given schema.
    """
    conn = connect_nba_db()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SET search_path TO {}, public;").format(sql.Identifier(schema))
        )
    conn.commit()
    return conn


def connect_games_db():
    return connect_schema_db(get_schema_name_games())


def connect_players_db():
    return connect_schema_db(get_schema_name_players())


def connect_odds_db():
    return connect_schema_db(get_schema_name_odds())


def connect_predictions_db():
    return connect_schema_db(get_schema_name_predictions())
