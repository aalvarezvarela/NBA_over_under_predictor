from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.fetch_odds_data.odds_mgm.get_odds_mgm import (
    build_mgm_odds_df_from_events,
    load_events_from_json,
)
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_games,
    get_schema_name_odds_mgm,
)
from psycopg import sql


def schema_exists(schema_name: str = None) -> bool:
    """Check if the schema exists in the database."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_odds_mgm()

        conn = connect_nba_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            exists = cur.fetchone()
        conn.close()
        return exists is not None
    except Exception as e:
        print(f"Error checking schema existence: {e}")
        return False


def create_odds_mgm_schema_if_not_exists(conn: psycopg.Connection, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_odds_mgm_table(drop_existing: bool = False):
    """Create the nba_odds_mgm table inside schema SCHEMA_NAME_ODDS_MGM."""
    try:
        schema = get_schema_name_odds_mgm()
        table = schema
        conn = connect_nba_db()

        create_odds_mgm_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    game_id TEXT NOT NULL,
                    game_date DATE NOT NULL,
                    game_date_captured TIMESTAMP WITH TIME ZONE NOT NULL,
                    team_home VARCHAR(100) NOT NULL,
                    team_away VARCHAR(100) NOT NULL,
                    team_home_original VARCHAR(100) NOT NULL,
                    team_away_original VARCHAR(100) NOT NULL,
                    season_year INTEGER NOT NULL,
                    mgm_total_line NUMERIC(8, 4) NOT NULL CHECK (mgm_total_line > 150),
                    mgm_moneyline_home NUMERIC(10, 4) NOT NULL,
                    mgm_moneyline_away NUMERIC(10, 4) NOT NULL,
                    mgm_spread_home NUMERIC(8, 4) NOT NULL,
                    mgm_spread_away NUMERIC(8, 4) NOT NULL,
                    mgm_total_over_money NUMERIC(8, 4),
                    mgm_total_under_money NUMERIC(8, 4)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{}(game_date, team_home, team_away)"
                ).format(
                    sql.Identifier("idx_odds_unique_game"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_id)").format(
                    sql.Identifier("idx_odds_game_id"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_odds_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date_captured)").format(
                    sql.Identifier("idx_odds_game_date_captured"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            
        conn.commit()
        conn.close()
        print(f"Table '{schema}.{table}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating odds table: {e}")
        return False


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    dup_names = df.columns[df.columns.duplicated()].unique()
    for name in dup_names:
        cols = df.loc[:, df.columns == name]
        df[name] = cols.bfill(axis=1).iloc[:, 0]
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def iter_json_paths(json_dir: str | Path) -> list[Path]:
    json_dir = Path(json_dir)
    return sorted(p for p in json_dir.glob("*.json") if p.is_file())


def load_odds_mgm_df_from_json_dir(json_dir: str | Path) -> pd.DataFrame:
    json_paths = iter_json_paths(json_dir)
    if not json_paths:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for json_path in json_paths:
        events = load_events_from_json(json_path)
        df = build_mgm_odds_df_from_events(events)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["game_date_captured"] = pd.to_datetime(
        df_all["game_date_captured"], utc=True, errors="coerce"
    )
    df_all["game_date"] = pd.to_datetime(
        df_all["game_date"], errors="coerce"
    ).dt.date

    df_all["team_home"] = df_all["team_home"].map(
        TEAM_NAME_STANDARDIZATION
    ).fillna(df_all["team_home"])
    df_all["team_away"] = df_all["team_away"].map(
        TEAM_NAME_STANDARDIZATION
    ).fillna(df_all["team_away"])

    df_all = df_all.dropna(
        subset=["game_date_captured", "game_date", "team_home", "team_away"]
    )
    df_all = df_all.drop_duplicates(
        subset=["game_date", "team_home", "team_away"], keep="last"
    )

    return df_all.reset_index(drop=True)


def load_games_for_season_years(season_years: Iterable[int]) -> pd.DataFrame:
    schema = get_schema_name_games()
    table = schema
    season_years = sorted({int(y) for y in season_years})

    conn = connect_nba_db()
    try:
        query_obj = sql.SQL(
            """
            SELECT game_id, game_date, team_name, home, season_year, season_type
            FROM {}.{}
            WHERE season_year = ANY(%s)
            ORDER BY game_date ASC
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))
        query = query_obj.as_string(conn)
        df = pd.read_sql_query(query, conn, params=(season_years,))

        if "season_type" in df.columns:
            season_type = df["season_type"].astype("string").str.lower()
            mask_pre = season_type.str.contains("pre", na=False)
            mask_allstar = season_type.str.contains("all", na=False)
            df = df[~(mask_pre | mask_allstar)]

        return df
    finally:
        conn.close()


def build_games_home_away_df(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.copy()
    df["team_name"] = df["team_name"].map(TEAM_NAME_STANDARDIZATION).fillna(
        df["team_name"]
    )
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    if "home" not in df:
        return pd.DataFrame(columns=["game_id", "game_date", "team_home", "team_away"])

    home = df[df["home"] == True]
    away = df[df["home"] == False]

    home = home.rename(columns={"team_name": "team_home"})[
        ["game_id", "game_date", "team_home"]
    ]
    away = away.rename(columns={"team_name": "team_away"})[
        ["game_id", "game_date", "team_away"]
    ]

    merged = home.merge(away, on=["game_id", "game_date"], how="inner")
    return merged


def merge_odds_with_games(odds_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    games_ha = build_games_home_away_df(games_df)
    odds_df = odds_df.copy()
   
    odds_df["game_date"] = pd.to_datetime(odds_df["game_date"], errors="coerce").dt.date

    merged = odds_df.merge(
        games_ha,
        left_on=["game_date", "team_home", "team_away"],
        right_on=["game_date", "team_home", "team_away"],
        how="left",
    )

    return merged


def upsert_odds_mgm_df(
    odds_df: pd.DataFrame, conn: psycopg.Connection | None = None
) -> int:
    if odds_df.empty:
        return 0

    schema = get_schema_name_odds_mgm()
    table = schema
    close_conn = False
    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    odds_df = odds_df.copy()
    odds_df["game_date_captured"] = pd.to_datetime(
        odds_df["game_date_captured"], utc=True, errors="coerce"
    )
    odds_df["game_date"] = pd.to_datetime(odds_df["game_date"], errors="coerce").dt.date

    odds_df = odds_df.dropna(
        subset=[
            "game_date",
            "game_date_captured",
            "team_home",
            "team_away",
            "mgm_total_line",
            "mgm_moneyline_home",
            "mgm_moneyline_away",
            "mgm_spread_home",
            "mgm_spread_away",
        ]
    )

    cols = [
        "game_id",
        "game_date",
        "game_date_captured",
        "team_home",
        "team_away",
        "team_home_original",
        "team_away_original",
        "season_year",
        "mgm_total_line",
        "mgm_moneyline_home",
        "mgm_moneyline_away",
        "mgm_spread_home",
        "mgm_spread_away",
        "mgm_total_over_money",
        "mgm_total_under_money",
    ]

    rows = [tuple(row) for row in odds_df[cols].itertuples(index=False, name=None)]

    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            game_id,
            game_date,
            game_date_captured,
            team_home,
            team_away,
            team_home_original,
            team_away_original,
            season_year,
            mgm_total_line,
            mgm_moneyline_home,
            mgm_moneyline_away,
            mgm_spread_home,
            mgm_spread_away,
            mgm_total_over_money,
            mgm_total_under_money
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (game_id)
        DO NOTHING
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, rows)
        conn.commit()
        return len(rows)
    finally:
        if close_conn:
            conn.close()


def get_recent_odds(limit: int = 10):
    """Retrieve the most recent odds data from schema SCHEMA_NAME_ODDS."""
    schema = get_schema_name_odds_mgm()
    table = schema
    try:
        conn = connect_nba_db()
        with conn.cursor() as cur:
            query = sql.SQL(
                "SELECT * FROM {}.{} ORDER BY game_date DESC LIMIT %s"
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(query, (limit,))
            rows = cur.fetchall()

            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in rows]

        conn.close()
        return results
    except Exception as e:
        print(f"Error retrieving recent odds: {e}")
        return []


def get_missing_bets_dates(
    games_df: pd.DataFrame, odds_df: pd.DataFrame
) -> list[str]:
    if games_df.empty:
        return []

    games_dates = pd.to_datetime(games_df["game_date"], errors="coerce").dt.date
    odds_dates = pd.to_datetime(odds_df["game_date"], errors="coerce").dt.date

    missing = sorted(set(games_dates.dropna()) - set(odds_dates.dropna()))
    return [d.isoformat() for d in missing]


def build_and_load_odds_mgm(
    json_dir: str | Path,
    season_years: Iterable[int],
) -> dict[str, object]:
    odds_df = load_odds_mgm_df_from_json_dir(json_dir)
    games_df = load_games_for_season_years(season_years)
    merged_df = merge_odds_with_games(odds_df, games_df)
    assert len(merged_df) == len(odds_df), "Merged df row count mismatch"
    inserted = upsert_odds_mgm_df(merged_df)
    missing_dates = get_missing_bets_dates(games_df, merged_df)

    return {
        "odds_rows": len(odds_df),
        "merged_rows": len(merged_df),
        "inserted_rows": inserted,
        "missing_dates": missing_dates,
    }


if __name__ == "__main__":
    json_dir = Path("/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/TheRundownApi_data")
    season_years = [2022, 2023, 2024, 2025]

    print("\nStep 1: Creating schema + nba_odds_mgm table...")
    if not create_odds_mgm_table():
        raise SystemExit(1)

    print("\nStep 2: Building odds df and merging with games...")
    results = build_and_load_odds_mgm(json_dir=json_dir, season_years=season_years)
    print(f"Loaded odds rows: {results['odds_rows']}")
    print(f"Merged odds rows: {results['merged_rows']}")
    print(f"Inserted/updated rows: {results['inserted_rows']}")

    missing_dates = results["missing_dates"]
    print(f"Missing bet dates ({len(missing_dates)}):")
    for d in missing_dates:
        print(f"  {d}")
