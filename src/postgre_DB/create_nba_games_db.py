import pandas as pd
import psycopg
from psycopg import sql

from .db_config import (
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

            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    season_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    team_id TEXT NOT NULL,
                    team_abbreviation VARCHAR(10),
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
            ).format(sql.Identifier(schema), sql.Identifier("nba_games"))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_nba_games_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier("nba_games"),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_id)").format(
                    sql.Identifier("idx_nba_games_team_id"),
                    sql.Identifier(schema),
                    sql.Identifier("nba_games"),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(season_id)").format(
                    sql.Identifier("idx_nba_games_season_id"),
                    sql.Identifier(schema),
                    sql.Identifier("nba_games"),
                )
            )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.nba_games' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating table: {e}")
        return False


def load_games_data_to_db(df: pd.DataFrame, conn: psycopg.Connection | None = None):
    """Load dataframe into schema SCHEMA_NAME_GAMES table nba_games."""
    close_conn = False
    schema = get_schema_name_games()

    try:
        if "teamSlug" in df.columns:
            df = df.drop(columns=["teamSlug"])

        if conn is None:
            conn = connect_nba_db()
            close_conn = True

        create_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if "SEASON_ID" in df.columns:
                df["SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)

            if "WL" in df.columns:
                df["WL"] = df["WL"].replace("", None)
                df["WL"] = df["WL"].where(df["WL"].notna(), None)

            if "HOME" in df.columns:
                df["HOME"] = df["HOME"].map(
                    {"True": True, "False": False, True: True, False: False}
                )

            if "GAME_DATE" in df.columns:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

            integer_cols = [
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
            ]
            for col in integer_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

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
                "PLUS_MINUS",
            ]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.where(pd.notna(df), None)

            columns = [col for col in df.columns if col != "teamSlug"]
            column_names = ", ".join([col.lower() for col in columns])
            placeholders = ", ".join(["%s"] * len(columns))

            insert_query = sql.SQL(
                """
                INSERT INTO {}.{} ({})
                VALUES ({})
                ON CONFLICT (game_id, team_id, season_id, season_year) DO NOTHING
                """
            ).format(
                sql.Identifier(schema),
                sql.Identifier("nba_games"),
                sql.SQL(column_names),
                sql.SQL(placeholders),
            )

            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                values = [tuple(row) for row in batch[columns].values]
                cur.executemany(insert_query, values)
                conn.commit()

            # Verify count
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier("nba_games")
                )
            )
            count = cur.fetchone()[0]
            print(f"Total rows in {schema}.nba_games: {count}")

        if close_conn:
            conn.close()
        return True

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback

        traceback.print_exc()
        if close_conn and conn:
            conn.close()
        return False


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
