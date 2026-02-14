from collections.abc import Iterable

import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_sportsbook,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_cleaned_games_for_odds,
)
from psycopg import sql


def load_games_for_sportsbook_update(
    season_year: str | int | None = None,
) -> pd.DataFrame:
    """Load cleaned games from DB (excludes All Star/Preseason, fixes home/away issues)."""
    games_df = load_cleaned_games_for_odds(season_year=season_year)
    if games_df.empty:
        return games_df

    games_df = games_df.copy()
    games_df["game_id"] = games_df["game_id"].astype(str)
    games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce")

    return games_df


def select_target_game_ids(
    games_df: pd.DataFrame,
    *,
    last_n_games: int | None = None,
) -> list[str]:
    """Return unique target GAME_ID values, optionally restricted to latest N games."""
    if games_df.empty:
        return []

    base = games_df[["game_id", "game_date"]].dropna(subset=["game_id"]).copy()
    base["game_id"] = base["game_id"].astype(str)
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")

    base = (
        base.sort_values(["game_date", "game_id"], ascending=[False, False])
        .drop_duplicates(subset=["game_id"], keep="first")
        .reset_index(drop=True)
    )

    if last_n_games is not None:
        if last_n_games <= 0:
            return []
        base = base.head(last_n_games)

    return base["game_id"].tolist()


def get_existing_sportsbook_game_ids(
    game_ids: Iterable[str] | None = None,
) -> set[str]:
    """Return game IDs that already exist in odds_sportsbook table."""
    schema = get_schema_name_odds_sportsbook()
    table = schema

    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            if game_ids is None:
                query = sql.SQL("SELECT DISTINCT game_id FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
                cur.execute(query)
            else:
                game_ids_list = [str(x) for x in game_ids]
                if not game_ids_list:
                    return set()
                query = sql.SQL(
                    "SELECT DISTINCT game_id FROM {}.{} WHERE game_id = ANY(%s)"
                ).format(sql.Identifier(schema), sql.Identifier(table))
                cur.execute(query, (game_ids_list,))

            rows = cur.fetchall()
            return {str(row[0]) for row in rows}
    finally:
        conn.close()


def get_missing_game_ids_to_scrape(
    target_game_ids: Iterable[str],
) -> list[str]:
    target_ids = [str(gid) for gid in target_game_ids]
    existing = get_existing_sportsbook_game_ids(target_ids)
    return [gid for gid in target_ids if gid not in existing]


def get_dates_for_game_ids(
    games_df: pd.DataFrame, game_ids: Iterable[str]
) -> list[pd.Timestamp]:
    if games_df.empty:
        return []

    game_ids_set = set(str(gid) for gid in game_ids)
    if not game_ids_set:
        return []

    base = games_df[["game_id", "game_date"]].copy()
    base["game_id"] = base["game_id"].astype(str)
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")

    out = (
        base[base["game_id"].isin(game_ids_set)]
        .dropna(subset=["game_date"])["game_date"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    return out.tolist()
