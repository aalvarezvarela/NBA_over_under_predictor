from pathlib import Path

import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_sportsbook_matchup_history,
)
from nba_ou.postgre_db.sportsbook_matchup_history.process_sportsbook_matchup_history_data import (
    MATCHUP_HISTORY_NUMERIC_COLUMNS,
    MATCHUP_HISTORY_TEXT_COLUMNS,
    build_matchup_history_df_from_csvs,
    load_games_for_matchup_history_creation,
    merge_matchup_history_with_games,
)
from psycopg import sql


def _matchup_history_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("game_id", "TEXT NOT NULL"),
        ("game_date", "DATE NOT NULL"),
        ("season_year", "INTEGER NOT NULL"),
        ("event_id", "TEXT NOT NULL"),
        ("start_time", "TEXT"),
        ("game_start_timestamp", "TIMESTAMP"),
        ("team_away", "VARCHAR(100) NOT NULL"),
        ("team_home", "VARCHAR(100) NOT NULL"),
    ]

    columns.extend((col, "NUMERIC(10, 4)") for col in MATCHUP_HISTORY_NUMERIC_COLUMNS)
    columns.extend((col, "TEXT") for col in MATCHUP_HISTORY_TEXT_COLUMNS)
    return columns


def _matchup_history_insert_columns() -> list[str]:
    return [name for name, _ in _matchup_history_columns()]


def create_sportsbook_matchup_history_schema_if_not_exists(
    conn: psycopg.Connection,
    schema: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_sportsbook_matchup_history_table(drop_existing: bool = False) -> bool:
    schema = get_schema_name_sportsbook_matchup_history()
    table = schema
    conn = connect_nba_db()

    try:
        create_sportsbook_matchup_history_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            column_defs = _matchup_history_columns()
            column_sql = ",\n".join(
                [f"{name} {dtype}" for name, dtype in column_defs]
                + [
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "PRIMARY KEY (game_id)",
                    "UNIQUE (event_id)",
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
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS game_start_timestamp TIMESTAMP"
                ).format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {}.{} DROP COLUMN IF EXISTS matchup_url").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )

            index_specs = [
                (f"idx_{schema}_game_date", ["game_date"]),
                (f"idx_{schema}_season_year", ["season_year"]),
                (f"idx_{schema}_game_lookup", ["game_id", "game_date", "season_year"]),
                (f"idx_{schema}_event_id", ["event_id"]),
                (f"idx_{schema}_teams", ["team_home", "team_away"]),
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
    except Exception as e:
        conn.rollback()
        print(f"Error creating sportsbook matchup-history table: {e}")
        return False
    finally:
        conn.close()


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out


def _prepare_matchup_history_for_upsert(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["game_id"] = out["game_id"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype(
        "Int64"
    )
    if "game_start_timestamp" in out:
        out["game_start_timestamp"] = pd.to_datetime(
            out["game_start_timestamp"], errors="coerce"
        )
    out["event_id"] = out["event_id"].astype(str)

    required = [
        "game_id",
        "game_date",
        "season_year",
        "event_id",
        "team_home",
        "team_away",
    ]
    before_drop = len(out)
    out = out.dropna(subset=required)
    dropped = before_drop - len(out)
    if dropped:
        print(f"Dropped {dropped} matchup-history rows missing required DB fields.")

    before_dedup = len(out)
    out = out.sort_values(["game_date", "game_id", "event_id"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    deduped = before_dedup - len(out)
    if deduped:
        print(f"Dropped {deduped} duplicate matchup-history rows on game_id.")

    before_event_dedup = len(out)
    out = out.drop_duplicates(subset=["event_id"], keep="last")
    event_deduped = before_event_dedup - len(out)
    if event_deduped:
        print(f"Dropped {event_deduped} duplicate matchup-history rows on event_id.")

    return _ensure_columns(out, _matchup_history_insert_columns())


def upsert_sportsbook_matchup_history_df(
    matchup_history_df: pd.DataFrame,
    conn: psycopg.Connection | None = None,
) -> int:
    if matchup_history_df.empty:
        return 0

    schema = get_schema_name_sportsbook_matchup_history()
    table = schema
    close_conn = False
    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    upload_df = _prepare_matchup_history_for_upsert(matchup_history_df)
    if upload_df.empty:
        if close_conn:
            conn.close()
        return 0

    cols = _matchup_history_insert_columns()
    upload_df = upload_df.astype(object).where(pd.notna(upload_df), None)
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
    finally:
        if close_conn:
            conn.close()


def build_and_load_sportsbook_matchup_history_from_csvs(
    matchup_history_root_dir: str | Path,
    *,
    season_dir_glob: str = "*",
    drop_existing: bool = False,
    strict_game_id_match: bool = False,
) -> dict[str, int]:
    if not create_sportsbook_matchup_history_table(drop_existing=drop_existing):
        raise RuntimeError(
            "Failed to create/validate sportsbook matchup-history table."
        )

    matchup_history_df = build_matchup_history_df_from_csvs(
        matchup_history_root_dir,
        season_dir_glob=season_dir_glob,
    )
    games_df = load_games_for_matchup_history_creation()
    merged_df = merge_matchup_history_with_games(matchup_history_df, games_df)

    null_game_id_count = (
        int(merged_df["game_id"].isna().sum())
        if "game_id" in merged_df
        else len(merged_df)
    )
    print(f"Number of matchup-history rows with null game_id: {null_game_id_count}")
    if strict_game_id_match and null_game_id_count:
        raise RuntimeError(
            f"{null_game_id_count} matchup-history rows could not be matched to game_id."
        )

    mapped_df = (
        merged_df.dropna(subset=["game_id"])
        if "game_id" in merged_df
        else pd.DataFrame()
    )
    inserted = upsert_sportsbook_matchup_history_df(mapped_df)

    return {
        "csv_rows": len(matchup_history_df),
        "merged_rows": len(merged_df),
        "unmatched_rows": null_game_id_count,
        "loaded_rows": inserted,
    }
