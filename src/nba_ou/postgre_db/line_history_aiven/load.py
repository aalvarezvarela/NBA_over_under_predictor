"""Bulk-load transformed line-history rows into Aiven.

Uses ``COPY`` into an ``UNLOGGED`` staging table and then a single
``INSERT ... SELECT ... ON CONFLICT DO NOTHING`` into the partitioned target.
``executemany`` would issue one round trip per row, which is untenable for ~1.9M
rows over a WAN; staging UNLOGGED also keeps the copy itself out of the WAL,
which matters when the 1 GB cap covers WAL too.
"""

from __future__ import annotations

import json

import pandas as pd
import psycopg
from psycopg import sql

from .schema import SCHEMA
from .transform import OUTPUT_COLUMNS

STAGING_TABLE = "lh_line_staging"

GAME_DIM_INSERT_COLUMNS = [
    "game_id",
    "game_date",
    "season_year",
    "tipoff_utc",
    "event_id",
    "team_home",
    "team_away",
]


def _to_native(value):
    """pandas NA/NaT -> None, numpy scalars -> Python scalars."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def upsert_games(conn: psycopg.Connection, game_dim: pd.DataFrame) -> int:
    if game_dim.empty:
        return 0

    rows = [
        tuple(_to_native(v) for v in row)
        for row in game_dim[GAME_DIM_INSERT_COLUMNS].itertuples(index=False, name=None)
    ]
    query = sql.SQL(
        """
        INSERT INTO {}.lh_game ({cols}) VALUES ({vals})
        ON CONFLICT (game_id) DO UPDATE SET
            tipoff_utc = EXCLUDED.tipoff_utc,
            game_date  = EXCLUDED.game_date
        """
    ).format(
        sql.Identifier(SCHEMA),
        cols=sql.SQL(", ").join(map(sql.Identifier, GAME_DIM_INSERT_COLUMNS)),
        vals=sql.SQL(", ").join(sql.Placeholder() * len(GAME_DIM_INSERT_COLUMNS)),
    )
    with conn.cursor() as cur:
        cur.executemany(query, rows)
    conn.commit()
    return len(rows)


def _create_staging(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(SCHEMA), sql.Identifier(STAGING_TABLE)
            )
        )
        cur.execute(
            sql.SQL(
                "CREATE UNLOGGED TABLE {}.{} (LIKE {}.lh_line INCLUDING DEFAULTS)"
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(STAGING_TABLE),
                sql.Identifier(SCHEMA),
            )
        )
    conn.commit()


def copy_rows(conn: psycopg.Connection, rows: pd.DataFrame) -> int:
    """COPY ``rows`` into the staging table."""
    _create_staging(conn)

    copy_sql = sql.SQL("COPY {}.{} ({cols}) FROM STDIN").format(
        sql.Identifier(SCHEMA),
        sql.Identifier(STAGING_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, OUTPUT_COLUMNS)),
    )

    written = 0
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            for row in rows[OUTPUT_COLUMNS].itertuples(index=False, name=None):
                copy.write_row(tuple(_to_native(v) for v in row))
                written += 1
    conn.commit()
    return written


def merge_staging(conn: psycopg.Connection, season_year: int) -> int:
    """Move staged rows into the partitioned table, ignoring existing keys."""
    query = sql.SQL(
        """
        INSERT INTO {schema}.lh_line ({cols})
        SELECT {cols} FROM {schema}.{staging}
        ON CONFLICT (game_id, market_id, book_id, line_ts, season_year) DO NOTHING
        """
    ).format(
        schema=sql.Identifier(SCHEMA),
        staging=sql.Identifier(STAGING_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, OUTPUT_COLUMNS)),
    )
    with conn.cursor() as cur:
        cur.execute(query)
        inserted = cur.rowcount
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(SCHEMA), sql.Identifier(STAGING_TABLE)
            )
        )
    conn.commit()
    return inserted


def record_load(
    conn: psycopg.Connection,
    *,
    season_year: int,
    timezone: str,
    confidence: str,
    source_rows: int,
    loaded_rows: int,
    dropped: dict[str, int],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {}.lh_load_meta
                    (season_year, timezone, confidence, source_rows,
                     loaded_rows, dropped_rows, loaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (season_year) DO UPDATE SET
                    timezone = EXCLUDED.timezone,
                    confidence = EXCLUDED.confidence,
                    source_rows = EXCLUDED.source_rows,
                    loaded_rows = EXCLUDED.loaded_rows,
                    dropped_rows = EXCLUDED.dropped_rows,
                    loaded_at = EXCLUDED.loaded_at
                """
            ).format(sql.Identifier(SCHEMA)),
            (
                season_year,
                timezone,
                confidence,
                source_rows,
                loaded_rows,
                json.dumps(dropped),
            ),
        )
    conn.commit()


def vacuum_analyze(conn: psycopg.Connection, season_year: int) -> None:
    from .schema import partition_name

    previous = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("VACUUM ANALYZE {}.{}").format(
                    sql.Identifier(SCHEMA), sql.Identifier(partition_name(season_year))
                )
            )
    finally:
        conn.autocommit = previous


def database_size(conn: psycopg.Connection) -> tuple[int, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_database_size(current_database()), "
            "pg_size_pretty(pg_database_size(current_database()))"
        )
        return cur.fetchone()
