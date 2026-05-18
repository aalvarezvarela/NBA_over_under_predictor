"""PostgreSQL database utilities for NBA data."""

__all__ = [
    "load_all_nba_data_from_db",
    "load_games_from_db",
    "load_players_from_db",
]


def __getattr__(name: str):
    """Load dataframe helpers lazily so config-only imports stay lightweight."""
    if name in __all__:
        from .config import db_loader

        return getattr(db_loader, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
