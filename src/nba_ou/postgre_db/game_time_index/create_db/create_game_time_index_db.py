from __future__ import annotations

import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_game_time_index,
)
from psycopg import sql
from psycopg.types.json import Jsonb

GAME_TIME_INDEX_SOURCE = "nba_cdn_boxscore"


def _game_time_index_columns() -> list[tuple[str, str]]:
    return [
        ("game_id", "TEXT NOT NULL"),
        ("game_date", "DATE NOT NULL"),
        ("season_year", "INTEGER NOT NULL"),
        ("game_time_local", "TEXT"),
        ("game_time_utc", "TIMESTAMPTZ"),
        ("game_time_home", "TEXT"),
        ("game_time_away", "TEXT"),
        ("game_et", "TEXT"),
        ("duration", "INTEGER"),
        ("game_code", "TEXT"),
        ("game_status_text", "TEXT"),
        ("game_status", "INTEGER"),
        ("regulation_periods", "INTEGER"),
        ("period", "INTEGER"),
        ("game_clock", "TEXT"),
        ("attendance", "INTEGER"),
        ("sellout", "TEXT"),
        ("arena_id", "BIGINT"),
        ("arena_name", "TEXT"),
        ("arena_city", "TEXT"),
        ("arena_state", "TEXT"),
        ("arena_country", "TEXT"),
        ("arena_timezone", "TEXT"),
        ("home_team_id", "BIGINT"),
        ("home_team_name", "TEXT"),
        ("home_team_city", "TEXT"),
        ("home_team_tricode", "TEXT"),
        ("home_team_score", "INTEGER"),
        ("away_team_id", "BIGINT"),
        ("away_team_name", "TEXT"),
        ("away_team_city", "TEXT"),
        ("away_team_tricode", "TEXT"),
        ("away_team_score", "INTEGER"),
        ("home_team_statistics", "JSONB"),
        ("away_team_statistics", "JSONB"),
        ("game_payload", "JSONB"),
        ("source", "TEXT NOT NULL"),
    ]


def get_game_time_index_insert_columns() -> list[str]:
    return [name for name, _ in _game_time_index_columns()]


def create_game_time_index_schema_if_not_exists(
    conn: psycopg.Connection,
    schema: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_game_time_index_table(
    *,
    drop_existing: bool = False,
    conn: psycopg.Connection | None = None,
) -> bool:
    schema = get_schema_name_game_time_index()
    table = schema
    close_conn = False

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    try:
        create_game_time_index_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            column_defs = _game_time_index_columns()
            column_sql = ",\n".join(
                [f"{name} {dtype}" for name, dtype in column_defs]
                + [
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "PRIMARY KEY (game_id)",
                ]
            )

            cur.execute(
                sql.SQL(
                    f"""
                    CREATE TABLE IF NOT EXISTS {{}}.{{}} (
                        {column_sql}
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            for column_name, column_type in column_defs:
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} {}"
                    ).format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(column_name),
                        sql.SQL(column_type),
                    )
                )

            # Preserve the exact source strings for non-UTC time fields.
            # TIMESTAMPTZ normalizes offsets, which collapses these values into UTC.
            for column_name in [
                "game_time_local",
                "game_time_home",
                "game_time_away",
                "game_et",
            ]:
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {}.{} ALTER COLUMN {} TYPE TEXT USING {}::TEXT"
                    ).format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(column_name),
                        sql.Identifier(column_name),
                    )
                )

            index_specs = [
                (f"idx_{schema}_game_date", ["game_date"]),
                (f"idx_{schema}_season_year", ["season_year"]),
                (f"idx_{schema}_game_time_utc", ["game_time_utc"]),
                (f"idx_{schema}_season_date", ["season_year", "game_date"]),
                (f"idx_{schema}_game_status", ["game_status"]),
            ]
            for index_name, columns in index_specs:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}({})").format(
                        sql.Identifier(index_name),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.SQL(", ").join(map(sql.Identifier, columns)),
                    )
                )

        conn.commit()
        print(f"Table '{schema}.{table}' created successfully!")
        return True
    except Exception as exc:
        conn.rollback()
        print(f"Error creating game-time index table: {exc}")
        return False
    finally:
        if close_conn:
            conn.close()


def load_game_time_index_for_season(
    season_year: int,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    schema = get_schema_name_game_time_index()
    table = schema
    close_conn = False

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    try:
        query_obj = sql.SQL("""
            SELECT {cols}
            FROM {}.{}
            WHERE season_year = %s
            ORDER BY game_date, game_id
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            cols=sql.SQL(", ").join(
                map(sql.Identifier, get_game_time_index_insert_columns())
            ),
        )
        query = query_obj.as_string(conn)
        return pd.read_sql_query(query, conn, params=(season_year,))
    finally:
        if close_conn:
            conn.close()


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out


def _prepare_game_time_index_for_upsert(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["game_id"] = out["game_id"].astype(str).str.strip()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype(
        "Int64"
    )
    for col in ["game_time_local", "game_time_home", "game_time_away", "game_et"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .where(out[col].notna(), None)
                .map(lambda value: str(value).strip() if value is not None else None)
            )

    if "game_time_utc" in out.columns:
        out["game_time_utc"] = pd.to_datetime(
            out["game_time_utc"],
            errors="coerce",
            utc=True,
        )

    integer_cols = [
        "duration",
        "game_status",
        "regulation_periods",
        "period",
        "attendance",
        "arena_id",
        "home_team_id",
        "home_team_score",
        "away_team_id",
        "away_team_score",
    ]
    for col in integer_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    if "source" not in out.columns:
        out["source"] = GAME_TIME_INDEX_SOURCE

    out["source"] = out["source"].fillna(GAME_TIME_INDEX_SOURCE).astype(str).str.strip()
    out["source"] = out["source"].replace("", GAME_TIME_INDEX_SOURCE)

    before_drop = len(out)
    out = out.dropna(subset=["game_id", "game_date", "season_year"])
    dropped = before_drop - len(out)
    if dropped:
        print(f"Dropped {dropped} game-time rows missing required DB fields.")

    before_dedup = len(out)
    out = out.sort_values(["game_date", "game_id"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    deduped = before_dedup - len(out)
    if deduped:
        print(f"Dropped {deduped} duplicate game-time rows on game_id.")

    return _ensure_columns(out, get_game_time_index_insert_columns())


def upsert_game_time_index_df(
    game_time_df: pd.DataFrame,
    conn: psycopg.Connection | None = None,
) -> int:
    if game_time_df.empty:
        return 0

    schema = get_schema_name_game_time_index()
    table = schema
    close_conn = False

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    upload_df = _prepare_game_time_index_for_upsert(game_time_df)
    if upload_df.empty:
        if close_conn:
            conn.close()
        return 0

    cols = get_game_time_index_insert_columns()
    upload_df = upload_df.astype(object).where(pd.notna(upload_df), None)
    json_cols = ["home_team_statistics", "away_team_statistics", "game_payload"]
    for col in json_cols:
        if col in upload_df.columns:
            upload_df[col] = upload_df[col].map(
                lambda value: Jsonb(value)
                if isinstance(value, (dict, list))
                else None
            )

    rows = [tuple(row) for row in upload_df[cols].itertuples(index=False, name=None)]

    update_cols = [col for col in cols if col != "game_id"]

    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            {cols}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT (game_id)
        DO UPDATE SET
            {updates},
            updated_at = CURRENT_TIMESTAMP
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in cols),
        updates=sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(
                sql.Identifier(col),
                sql.Identifier(col),
            )
            for col in update_cols
        ),
    )

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, rows)
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        if close_conn:
            conn.close()
