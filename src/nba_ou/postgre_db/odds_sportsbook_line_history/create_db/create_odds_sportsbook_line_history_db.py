from pathlib import Path

import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_sportsbook_line_history_moneyline,
    get_schema_name_odds_sportsbook_line_history_spread,
    get_schema_name_odds_sportsbook_line_history_totals,
)
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
    build_line_history_df_from_csvs,
    load_games_for_line_history_creation,
    merge_line_history_with_games,
    normalize_line_history_market,
    normalize_line_history_markets,
)
from psycopg import sql

MARKET_SCHEMA_GETTERS = {
    MARKET_TOTALS: get_schema_name_odds_sportsbook_line_history_totals,
    MARKET_MONEYLINE: get_schema_name_odds_sportsbook_line_history_moneyline,
    MARKET_SPREAD: get_schema_name_odds_sportsbook_line_history_spread,
}


def get_line_history_schema_name(market: str) -> str:
    normalized_market = normalize_line_history_market(market)
    return MARKET_SCHEMA_GETTERS[normalized_market]()


def _line_history_columns() -> list[tuple[str, str]]:
    return [
        ("game_id", "TEXT NOT NULL"),
        ("game_date", "DATE NOT NULL"),
        ("season_year", "INTEGER NOT NULL"),
        ("event_id", "TEXT NOT NULL"),
        ("start_time", "TEXT"),
        ("game_start_timestamp_utc", "TIMESTAMPTZ"),
        ("team_away", "VARCHAR(100) NOT NULL"),
        ("team_home", "VARCHAR(100) NOT NULL"),
        ("bookmaker", "VARCHAR(100) NOT NULL"),
        ("bookmaker_slug", "VARCHAR(100) NOT NULL"),
        ("market", "VARCHAR(32) NOT NULL"),
        ("row_kind", "VARCHAR(32)"),
        ("change_order", "INTEGER"),
        ("timestamp_raw", "TEXT"),
        ("line_timestamp", "TIMESTAMP NOT NULL"),
        ("left_label", "TEXT"),
        ("right_label", "TEXT"),
        ("left_value_raw", "TEXT"),
        ("right_value_raw", "TEXT"),
        ("left_line", "NUMERIC(10, 4)"),
        ("left_price", "NUMERIC(10, 4)"),
        ("right_line", "NUMERIC(10, 4)"),
        ("right_price", "NUMERIC(10, 4)"),
    ]


def _line_history_insert_columns() -> list[str]:
    return [name for name, _ in _line_history_columns()]


def create_line_history_schema_if_not_exists(
    conn: psycopg.Connection,
    schema: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_odds_sportsbook_line_history_table(
    market: str,
    *,
    drop_existing: bool = False,
    conn: psycopg.Connection | None = None,
) -> bool:
    normalized_market = normalize_line_history_market(market)
    schema = get_line_history_schema_name(normalized_market)
    table = schema
    close_conn = False

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    try:
        create_line_history_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            column_defs = _line_history_columns()
            column_sql = ",\n".join(
                [f"{name} {dtype}" for name, dtype in column_defs]
                + [
                    "line_history_id BIGSERIAL PRIMARY KEY",
                    "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                    "UNIQUE (game_id, bookmaker_slug, line_timestamp)",
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
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS game_start_timestamp_utc TIMESTAMPTZ"
                ).format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = %s
                            AND table_name = %s
                            AND column_name = 'game_start_timestamp'
                    ) THEN
                        EXECUTE format(
                            'UPDATE %I.%I SET game_start_timestamp_utc = COALESCE(game_start_timestamp_utc, game_start_timestamp AT TIME ZONE ''UTC'')',
                            %s,
                            %s
                        );
                        EXECUTE format(
                            'ALTER TABLE %I.%I DROP COLUMN game_start_timestamp',
                            %s,
                            %s
                        );
                    END IF;
                END $$;
                """,
                (schema, table, schema, table, schema, table),
            )
            cur.execute(
                sql.SQL("ALTER TABLE {}.{} DROP COLUMN IF EXISTS matchup_url").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} DROP COLUMN IF EXISTS line_history_url"
                ).format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )

            index_specs = [
                (f"idx_{schema}_game_id", ["game_id"]),
                (f"idx_{schema}_game_date", ["game_date"]),
                (f"idx_{schema}_season_year", ["season_year"]),
                (f"idx_{schema}_game_lookup", ["game_id", "game_date", "season_year"]),
                (f"idx_{schema}_bookmaker", ["bookmaker_slug"]),
                (f"idx_{schema}_line_timestamp", ["line_timestamp"]),
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
        print(f"Error creating {normalized_market} line-history table: {e}")
        return False
    finally:
        if close_conn:
            conn.close()


def create_all_odds_sportsbook_line_history_tables(
    *,
    drop_existing: bool = False,
    markets: list[str] | None = None,
) -> bool:
    target_markets = normalize_line_history_markets(markets)
    conn = connect_nba_db()
    try:
        ok = True
        for market in target_markets:
            ok = (
                create_odds_sportsbook_line_history_table(
                    market,
                    drop_existing=drop_existing,
                    conn=conn,
                )
                and ok
            )
        return ok
    finally:
        conn.close()


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out


def _prepare_line_history_for_upsert(df: pd.DataFrame, market: str) -> pd.DataFrame:
    normalized_market = normalize_line_history_market(market)
    out = df.copy()
    out = out[out["market"] == normalized_market]
    if out.empty:
        return out

    out["game_id"] = out["game_id"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype(
        "Int64"
    )
    out["line_timestamp"] = pd.to_datetime(out["line_timestamp"], errors="coerce")
    if "game_start_timestamp_utc" in out:
        out["game_start_timestamp_utc"] = pd.to_datetime(
            out["game_start_timestamp_utc"], errors="coerce", utc=True
        )
    out["event_id"] = out["event_id"].astype(str)
    out["bookmaker_slug"] = out["bookmaker_slug"].astype(str).str.strip().str.lower()

    required = [
        "game_id",
        "game_date",
        "season_year",
        "event_id",
        "team_home",
        "team_away",
        "bookmaker",
        "bookmaker_slug",
        "market",
        "line_timestamp",
    ]
    before_drop = len(out)
    out = out.dropna(subset=required)
    dropped = before_drop - len(out)
    if dropped:
        print(f"Dropped {dropped} {normalized_market} rows missing required DB fields.")

    out = out.sort_values(
        ["game_id", "bookmaker_slug", "line_timestamp", "change_order"],
        kind="mergesort",
    )
    before_dedup = len(out)
    out = out.drop_duplicates(
        subset=["game_id", "bookmaker_slug", "line_timestamp"],
        keep="last",
    )
    deduped = before_dedup - len(out)
    if deduped:
        print(
            f"Dropped {deduped} duplicate {normalized_market} rows on "
            "game_id/bookmaker_slug/line_timestamp."
        )

    return _ensure_columns(out, _line_history_insert_columns())


def upsert_odds_sportsbook_line_history_df(
    line_history_df: pd.DataFrame,
    market: str,
    *,
    conn: psycopg.Connection | None = None,
) -> int:
    if line_history_df.empty:
        return 0

    normalized_market = normalize_line_history_market(market)
    schema = get_line_history_schema_name(normalized_market)
    table = schema
    close_conn = False
    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    upload_df = _prepare_line_history_for_upsert(line_history_df, normalized_market)
    if upload_df.empty:
        if close_conn:
            conn.close()
        return 0

    cols = _line_history_insert_columns()
    upload_df = upload_df.astype(object).where(pd.notna(upload_df), None)
    rows = [tuple(row) for row in upload_df[cols].itertuples(index=False, name=None)]

    update_cols = [
        col
        for col in cols
        if col not in {"game_id", "bookmaker_slug", "line_timestamp"}
    ]
    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            {cols}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT (game_id, bookmaker_slug, line_timestamp)
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


def build_and_load_odds_sportsbook_line_history_from_csvs(
    line_history_root_dir: str | Path,
    *,
    season_dir_glob: str = "*",
    markets: list[str] | None = None,
    drop_existing: bool = False,
    strict_game_id_match: bool = False,
) -> dict[str, object]:
    target_markets = normalize_line_history_markets(markets)
    if not create_all_odds_sportsbook_line_history_tables(
        drop_existing=drop_existing,
        markets=target_markets,
    ):
        raise RuntimeError("Failed to create/validate line-history tables.")

    line_history_df = build_line_history_df_from_csvs(
        line_history_root_dir,
        season_dir_glob=season_dir_glob,
        markets=target_markets,
    )
    games_df = load_games_for_line_history_creation()
    merged_df = merge_line_history_with_games(line_history_df, games_df)

    null_game_id_count = (
        int(merged_df["game_id"].isna().sum())
        if "game_id" in merged_df
        else len(merged_df)
    )
    print(f"Number of line-history rows with null game_id: {null_game_id_count}")
    if strict_game_id_match and null_game_id_count:
        raise RuntimeError(
            f"{null_game_id_count} line-history rows could not be matched to game_id."
        )

    mapped_df = (
        merged_df.dropna(subset=["game_id"])
        if "game_id" in merged_df
        else pd.DataFrame()
    )
    rows_by_market: dict[str, int] = {}

    conn = connect_nba_db()
    try:
        for market in target_markets:
            rows_by_market[market] = upsert_odds_sportsbook_line_history_df(
                mapped_df,
                market,
                conn=conn,
            )
    finally:
        conn.close()

    return {
        "csv_rows": len(line_history_df),
        "merged_rows": len(merged_df),
        "unmatched_rows": null_game_id_count,
        "loaded_rows_by_market": rows_by_market,
    }
