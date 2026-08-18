"""Schema for the Aiven-hosted SBR line-history store.

Deliberately narrow. The instance has a 1 GB cap covering heap, indexes *and*
WAL, so every column is either a join key, a filter, or a price:

* ``line``/``price`` are ``SMALLINT``. Lines are always half-points, so they are
  stored doubled (``224.5 -> 449``) which is exact and costs 2 bytes instead of
  ~12 for ``NUMERIC``. Prices are American odds and fit natively once the
  "off the board" sentinel is nulled out.
* Everything derivable from ``game_id`` (team names, matchup URLs) or from the
  parsed values (``timestamp_raw``, ``left_value_raw``) is dropped.
* ``lh_line`` is LIST-partitioned by season so a season can be dropped instantly
  if the cap is ever reached, and so season-filtered reads prune to one
  partition and need no secondary index.

``mins_to_tip`` and ``is_pregame`` are not conveniences: SBR records in-play
ticks with the same ``row_kind`` as pre-game ones, so they are the only thing
separating a legitimate feature row from target leakage. Both are NOT NULL.
"""

from __future__ import annotations

import psycopg
from psycopg import sql

SCHEMA = "line_history"

MARKETS: tuple[tuple[int, str], ...] = (
    (1, "totals"),
    (2, "point_spread"),
    (3, "money_line"),
)

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_book (
        book_id  SMALLINT PRIMARY KEY,
        slug     TEXT NOT NULL UNIQUE,
        name     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_market (
        market_id SMALLINT PRIMARY KEY,
        code      TEXT NOT NULL UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_game (
        game_id     TEXT PRIMARY KEY,
        game_date   DATE        NOT NULL,
        season_year SMALLINT    NOT NULL,
        tipoff_utc  TIMESTAMPTZ NOT NULL,
        event_id    INTEGER,
        team_home   TEXT,
        team_away   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_line (
        game_id     TEXT        NOT NULL,
        season_year SMALLINT    NOT NULL,
        market_id   SMALLINT    NOT NULL,
        book_id     SMALLINT    NOT NULL,
        line_ts     TIMESTAMPTZ NOT NULL,
        mins_to_tip INTEGER     NOT NULL,
        is_pregame  BOOLEAN     NOT NULL,
        is_opener   BOOLEAN     NOT NULL DEFAULT FALSE,
        left_line   SMALLINT,
        left_price  SMALLINT,
        right_line  SMALLINT,
        right_price SMALLINT,
        -- season_year is last only because Postgres requires the partition key
        -- inside any unique constraint; game_id stays the leading column so
        -- "all lines for game X" still uses the index prefix.
        PRIMARY KEY (game_id, market_id, book_id, line_ts, season_year)
    ) PARTITION BY LIST (season_year)
    """,
    # Records which timezone each season was loaded under, so the Phase 0
    # calibration stays auditable after the fact.
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_load_meta (
        season_year   SMALLINT PRIMARY KEY,
        timezone      TEXT NOT NULL,
        confidence    TEXT NOT NULL,
        source_rows   INTEGER,
        loaded_rows   INTEGER,
        dropped_rows  JSONB,
        loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)


def create_schema(conn: psycopg.Connection) -> None:
    """Create the schema, dimensions and the partitioned fact table."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        for statement in DDL_STATEMENTS:
            cur.execute(sql.SQL(statement.format(schema=SCHEMA)))

        for market_id, code in MARKETS:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.lh_market (market_id, code) VALUES (%s, %s) "
                    "ON CONFLICT (market_id) DO NOTHING"
                ).format(sql.Identifier(SCHEMA)),
                (market_id, code),
            )
    conn.commit()


def partition_name(season_year: int) -> str:
    return f"lh_line_{season_year}"


def create_season_partition(conn: psycopg.Connection, season_year: int) -> None:
    # Partition bounds are DDL and cannot be parameterised, so the (integer)
    # season is embedded as a literal.
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.{} "
                "PARTITION OF {}.lh_line FOR VALUES IN ({})"
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(partition_name(season_year)),
                sql.Identifier(SCHEMA),
                sql.Literal(int(season_year)),
            )
        )
    conn.commit()


def ensure_books(conn: psycopg.Connection, slugs: list[str]) -> dict[str, int]:
    """Register any unseen bookmakers and return the full slug -> id mapping.

    Ids are assigned in first-seen order rather than by SERIAL so that a reload
    from the same CSVs is reproducible.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT slug, book_id FROM {}.lh_book").format(
                sql.Identifier(SCHEMA)
            )
        )
        mapping = {slug: book_id for slug, book_id in cur.fetchall()}

        next_id = max(mapping.values(), default=0) + 1
        for slug in sorted(set(slugs)):
            if slug in mapping:
                continue
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.lh_book (book_id, slug, name) VALUES (%s, %s, %s) "
                    "ON CONFLICT (slug) DO NOTHING"
                ).format(sql.Identifier(SCHEMA)),
                (next_id, slug, slug.replace("_", " ").title()),
            )
            mapping[slug] = next_id
            next_id += 1
    conn.commit()
    return mapping


def market_ids() -> dict[str, int]:
    return {code: market_id for market_id, code in MARKETS}


def drop_schema(conn: psycopg.Connection) -> None:
    """Tear everything down. Used by ``--reset``."""
    # Callers may reach here from an except: block with a failed transaction.
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(SCHEMA))
        )
    conn.commit()
