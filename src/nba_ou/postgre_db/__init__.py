"""PostgreSQL database utilities for NBA data."""

__all__ = [
    "load_all_nba_data_from_db",
    "load_all_star_voting_from_db",
    "load_games_from_db",
    "load_players_from_db",
]


def __getattr__(name: str):
    """Load dataframe helpers lazily so config-only imports stay lightweight."""
    if name == "load_all_star_voting_from_db":
        from .all_star_voting.fetch_data_from_db import (
            load_all_star_voting_from_db,
        )

        return load_all_star_voting_from_db

    if name in {
        "load_all_nba_data_from_db",
        "load_games_from_db",
        "load_players_from_db",
    }:
        from .config import db_loader

        return getattr(db_loader, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
