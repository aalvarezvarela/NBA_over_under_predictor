import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from .config.db_config import (
    connect_nba_db,
    connect_postgres_db,
    get_db_credentials,
    get_schema_name_games,
)


def create_database():
    """Create the single PostgreSQL database (DB_NAME) if it doesn't exist."""
    try:
        db_name = get_db_credentials()["dbname"]

        conn = connect_postgres_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,)
            )
            exists = cur.fetchone()

            if not exists:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
                print(f"Database '{db_name}' created successfully!")
            else:
                print(f"Database '{db_name}' already exists.")

        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def create_schema_if_not_exists(conn: psycopg.Connection, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_games_schema(drop_existing: bool = True):
    """Create the nba_games table in schema SCHEMA_NAME_GAMES."""
    try:
        schema = get_schema_name_games()
        conn = connect_nba_db()

        create_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier("nba_games"),
                    )
                )

            create_table_query = sql.SQL(  # TODO: Fix HOME column to poorly calculated
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    season_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    team_id TEXT NOT NULL,
                    team_abbreviation VARCHAR(100),
                    team_name VARCHAR(100),
                    game_id TEXT NOT NULL,
                    game_date DATE NOT NULL,
                    matchup VARCHAR(50),
                    wl VARCHAR(1),
                    min INTEGER,
                    pts INTEGER NOT NULL,
                    fgm INTEGER,
                    fga INTEGER,
                    fg_pct NUMERIC(5, 3),
                    fg3m INTEGER,
                    fg3a INTEGER,
                    fg3_pct NUMERIC(5, 3),
                    ftm INTEGER,
                    fta INTEGER,
                    ft_pct NUMERIC(5, 3),
                    oreb INTEGER,
                    dreb INTEGER,
                    reb INTEGER,
                    ast INTEGER,
                    stl INTEGER,
                    blk INTEGER,
                    tov INTEGER,
                    pf INTEGER,
                    plus_minus INTEGER,
                    season_type VARCHAR(20),
                    home BOOLEAN,
                    team_city VARCHAR(50),
                    e_off_rating NUMERIC(8, 3),
                    off_rating NUMERIC(8, 3),
                    e_def_rating NUMERIC(8, 3),
                    def_rating NUMERIC(8, 3),
                    e_net_rating NUMERIC(8, 3),
                    net_rating NUMERIC(8, 3),
                    ast_pct NUMERIC(8, 3),
                    ast_tov NUMERIC(8, 3),
                    ast_ratio NUMERIC(8, 3),
                    oreb_pct NUMERIC(8, 3),
                    dreb_pct NUMERIC(8, 3),
                    reb_pct NUMERIC(8, 3),
                    e_tm_tov_pct NUMERIC(8, 3),
                    tm_tov_pct NUMERIC(8, 3),
                    efg_pct NUMERIC(8, 3),
                    ts_pct NUMERIC(8, 3),
                    usg_pct NUMERIC(8, 3),
                    e_usg_pct NUMERIC(8, 3),
                    e_pace NUMERIC(8, 3),
                    pace NUMERIC(8, 3),
                    pace_per40 NUMERIC(8, 3),
                    poss NUMERIC(8, 3),
                    pie NUMERIC(8, 3),
                    PRIMARY KEY (game_id, team_id, season_id, season_year)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(schema))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_nba_games_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_id)").format(
                    sql.Identifier("idx_nba_games_team_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(season_id)").format(
                    sql.Identifier("idx_nba_games_season_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.{schema}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating table: {e}")
        return False


def load_games_data_to_db(
    df: pd.DataFrame, conn: psycopg.Connection | None = None
) -> bool:
    close_conn = False
    schema = get_schema_name_games()

    if "teamSlug" in df.columns:
        df = df.drop(columns=["teamSlug"])

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    create_schema_if_not_exists(conn, schema)

    # 1) Create SEASON_YEAR
    if "SEASON_ID" in df.columns:
        df["SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype("Int64")

    # 2) Clean WL / HOME / GAME_DATE
    if "WL" in df.columns:
        df["WL"] = df["WL"].replace("", None)
        df["WL"] = df["WL"].where(df["WL"].notna(), None)

    if "HOME" in df.columns:
        df["HOME"] = df["HOME"].map(
            {"True": True, "False": False, True: True, False: False}
        )

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # 3) Enforce exact column set in the table
    insert_cols = [
        "SEASON_ID",
        "SEASON_YEAR",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "PTS",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PLUS_MINUS",
        "SEASON_TYPE",
        "HOME",
        "TEAM_CITY",
        "E_OFF_RATING",
        "OFF_RATING",
        "E_DEF_RATING",
        "DEF_RATING",
        "E_NET_RATING",
        "NET_RATING",
        "AST_PCT",
        "AST_TOV",
        "AST_RATIO",
        "OREB_PCT",
        "DREB_PCT",
        "REB_PCT",
        "E_TM_TOV_PCT",
        "TM_TOV_PCT",
        "EFG_PCT",
        "TS_PCT",
        "USG_PCT",
        "E_USG_PCT",
        "E_PACE",
        "PACE",
        "PACE_PER40",
        "POSS",
        "PIE",
    ]

    missing = [c for c in insert_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for insert: {missing}")

    df = df[insert_cols].copy()

    # 4) Cast INTEGER columns including PLUS_MINUS (table defines plus_minus INTEGER)
    integer_cols = [
        "SEASON_YEAR",
        "MIN",
        "PTS",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PLUS_MINUS",
    ]
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = [
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "E_OFF_RATING",
        "OFF_RATING",
        "E_DEF_RATING",
        "DEF_RATING",
        "E_NET_RATING",
        "NET_RATING",
        "AST_PCT",
        "AST_TOV",
        "AST_RATIO",
        "OREB_PCT",
        "DREB_PCT",
        "REB_PCT",
        "E_TM_TOV_PCT",
        "TM_TOV_PCT",
        "EFG_PCT",
        "TS_PCT",
        "USG_PCT",
        "E_USG_PCT",
        "E_PACE",
        "PACE",
        "PACE_PER40",
        "POSS",
        "PIE",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # TEAM_ID is TEXT in DB; enforce string to avoid ambiguous casts
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    df["PLUS_MINUS"] = df["PLUS_MINUS"].replace({np.nan: None})
    df["PLUS_MINUS"] = df["PLUS_MINUS"].replace(pd.NA, None)
    # NaN -> None for psycopg
    df = df.where(pd.notna(df), None)

    # 5) Build a safe INSERT with identifiers for each column
    col_idents = [sql.Identifier(c.lower()) for c in insert_cols]
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in insert_cols)

    insert_query = sql.SQL("""
        INSERT INTO {}.{} ({})
        VALUES ({})
        ON CONFLICT (game_id, team_id, season_id, season_year) DO NOTHING
    """).format(
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.SQL(", ").join(col_idents),
        placeholders,
    )

    batch_size = 1000
    with conn.cursor() as cur:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            values = [tuple(r) for r in batch.to_numpy()]
            try:
                cur.executemany(insert_query, values)
                conn.commit()
            except Exception as e:
                conn.rollback()

                # Print one representative row to debug quickly
                r0 = batch.iloc[0]
                print(f"Error inserting batch starting at index {i}: {e}")
                print(
                    "Example row keys:",
                    {
                        "GAME_ID": r0.get("GAME_ID"),
                        "TEAM_ID": r0.get("TEAM_ID"),
                        "SEASON_ID": r0.get("SEASON_ID"),
                        "SEASON_YEAR": r0.get("SEASON_YEAR"),
                    },
                )
                raise

        # Verify count
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                sql.Identifier(schema), sql.Identifier(schema)
            )
        )
        count = cur.fetchone()[0]
        print(f"Total rows in {schema}.{schema}: {count}")

    if close_conn:
        conn.close()

    return True


if __name__ == "__main__":
    print("Step 1: Creating database...")
    if not create_database():
        exit(1)

    print("\nStep 2: Creating table...")
    if not create_games_schema():
        exit(1)

    print("\nStep 3: Loading combined data...")
    # Load the combined dataframe
    from combine_games_data import combine_all_nba_games

    data_dir = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    )
    df_all_games = combine_all_nba_games(data_dir)

    if df_all_games is not None:
        print("\nStep 4: Inserting data into PostgreSQL...")
        load_games_data_to_db(df_all_games)
    else:
        print("Failed to load combined dataframe!")
        exit(1)

    print("\n✅ Database setup complete!")
