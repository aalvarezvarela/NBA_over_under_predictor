"""
Centralized database configuration module.
Reads database credentials from config.ini file.
"""

import configparser
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent

_CONFIG_INI = _PROJECT_ROOT / "config.ini"
_SECRETS_INI = _PROJECT_ROOT / "config.secrets.ini"


def get_config() -> configparser.ConfigParser:
    """Load configuration from config.ini and overlay config.secrets.ini if present."""
    if not _CONFIG_INI.exists():
        raise FileNotFoundError(f"Config file not found at: {_CONFIG_INI}")

    config = configparser.ConfigParser()
    config.read(_CONFIG_INI)

    # Overlay optional local secrets file (gitignored)
    if _SECRETS_INI.exists():
        config.read(_SECRETS_INI)

    return config


def _get_env_or_config(
    config: configparser.ConfigParser,
    section: str,
    key: str,
    env_var: str,
    *,
    required: bool = True,
    fallback: str | None = None,
) -> str:
    """Resolve value with priority: env var -> config -> fallback -> error."""
    v = os.getenv(env_var)
    if v is not None and v != "":
        return v

    if config.has_option(section, key):
        v = config.get(section, key).strip()
        if v != "":
            return v

    if fallback is not None:
        return fallback

    if required:
        raise ValueError(
            f"Missing required setting [{section}] {key}. "
            f"Set env var {env_var} or define it in config.secrets.ini."
        )

    return ""


def get_db_env() -> str:
    """Return current DB environment: local|supabase."""
    config = get_config()
    return (
        os.getenv("DB_ENV", config.get("Database", "DB_ENV", fallback="local"))
        .strip()
        .lower()
    )


def get_db_credentials() -> dict[str, Any]:
    """
    Get database credentials with env var overrides and secrets support.
    """
    config = get_config()
    db_env = get_db_env()
    section = "DatabaseSupabase" if db_env == "supabase" else "DatabaseLocal"

    # Non-secret settings (config.ini)
    user = config.get(section, "DB_USER")
    host = config.get(section, "DB_HOST")
    port = config.get(section, "DB_PORT")
    sslmode = config.get(section, "DB_SSLMODE", fallback="").strip() or None

    # DB name can be per-environment; fallback to [Database].DB_NAME
    dbname = config.get(section, "DB_NAME", fallback=config.get("Database", "DB_NAME"))
    dbname = dbname.strip().strip('"').strip("'")

    # Secret password: env var preferred, else secrets file
    password_env = (
        "SUPABASE_DB_PASSWORD" if db_env == "supabase" else "LOCAL_DB_PASSWORD"
    )
    password = _get_env_or_config(
        config, section, "DB_PASSWORD", password_env, required=True
    )

    return {
        "env": db_env,
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "dbname": dbname,
        "sslmode": sslmode,
    }


def get_schema_name_games() -> str:
    return get_config().get("Database", "SCHEMA_NAME_GAMES")


def get_schema_name_players() -> str:
    return get_config().get("Database", "SCHEMA_NAME_PLAYERS")


def get_schema_name_odds() -> str:
    return get_config().get("Database", "SCHEMA_NAME_ODDS")


def get_schema_name_predictions() -> str:
    return get_config().get("Database", "SCHEMA_NAME_PREDICTIONS")


def connect_postgres_db() -> psycopg.Connection:
    """
    Connect to the 'postgres' database for admin tasks.

    Note: On Supabase you typically do not need admin DB creation; still usable for
    maintenance queries if your role permits.
    """
    c = get_db_credentials()

    kwargs: dict[str, Any] = dict(
        dbname="postgres",
        user=c["user"],
        password=c["password"],
        host=c["host"],
        port=c["port"],
        autocommit=True,
    )
    if c["sslmode"]:
        kwargs["sslmode"] = c["sslmode"]

    return psycopg.connect(**kwargs)


def connect_nba_db() -> psycopg.Connection:
    """Connect to the configured database (local or Supabase)."""
    c = get_db_credentials()
    print(f"Connecting to database environment: {c['env']}")

    kwargs: dict[str, Any] = dict(
        dbname=c["dbname"],
        user=c["user"],
        password=c["password"],
        host=c["host"],
        port=c["port"],
    )
    if c["sslmode"]:
        kwargs["sslmode"] = c["sslmode"]  # supabase: "require"

    return psycopg.connect(**kwargs)


def connect_schema_db(schema: str) -> psycopg.Connection:
    """Connect and set search_path to the given schema."""
    conn = connect_nba_db()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SET search_path TO {}, public;").format(sql.Identifier(schema))
        )
    conn.commit()
    return conn


def connect_games_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_games())


def connect_players_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_players())


def connect_odds_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_odds())


def connect_predictions_db() -> psycopg.Connection:
    return connect_schema_db(get_schema_name_predictions())
