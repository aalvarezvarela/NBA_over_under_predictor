import pandas as pd
import psycopg
from psycopg import sql

from .config.db_config import (
    connect_nba_db,
    connect_postgres_db,
    get_db_credentials,
    get_schema_name_players,
)


def create_players_schema_database():
    """
    Create the single PostgreSQL database (DB_NAME from config) if it doesn't exist.
    You can run this once globally, but keeping it here is fine.
    """
    try:
        db_name = get_db_credentials()["dbname"]

        conn = connect_postgres_db()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
            exists = cur.fetchone()

            if not exists:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
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
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    conn.commit()


def create_players_table(drop_existing: bool = True):
    """Create the nba_players table inside schema SCHEMA_NAME_PLAYERS."""
    try:
        schema = get_schema_name_players()
        conn = connect_nba_db()

        create_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(schema),
                    )
                )

            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    game_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    team_id TEXT NOT NULL,
                    team_abbreviation VARCHAR(100),
                    team_city VARCHAR(100),
                    team_name VARCHAR(100),
                    player_id TEXT NOT NULL,
                    player_name VARCHAR(100),
                    nickname VARCHAR(100),
                    firstname VARCHAR(100),
                    familyname VARCHAR(100),
                    start_position VARCHAR(100),
                    comment TEXT,
                    jerseynum VARCHAR(10),
                    min VARCHAR(20),
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
                    pts INTEGER,
                    plus_minus NUMERIC(8, 3),
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
                    PRIMARY KEY (game_id, team_id, player_id, season_year)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(schema))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_id)").format(
                    sql.Identifier("idx_player_game_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_id)").format(
                    sql.Identifier("idx_player_team_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(player_id)").format(
                    sql.Identifier("idx_player_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(player_name)").format(
                    sql.Identifier("idx_player_name"),
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


def load_players_data_to_db(df: pd.DataFrame, conn: psycopg.Connection | None = None):
    """Load players dataframe into schema SCHEMA_NAME_PLAYERS table nba_players."""
    close_conn = False
    schema = get_schema_name_players()

    # Remove unnecessary slug columns
    columns_to_remove = ["teamSlug", "playerSlug"]
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed '{col}' column")

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    create_schema_if_not_exists(conn, schema)

    with conn.cursor() as cur:
        print("Converting data types...")

        integer_cols = [
            "FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB",
            "AST","STL","BLK","TOV","PF","PTS",
        ]
        for col in integer_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                df[col] = df[col].astype(object).where(df[col].notna(), None)

        float_cols = [
            "FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS","E_OFF_RATING","OFF_RATING",
            "E_DEF_RATING","DEF_RATING","E_NET_RATING","NET_RATING","AST_PCT",
            "AST_TOV","AST_RATIO","OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT",
            "EFG_PCT","TS_PCT","USG_PCT","E_USG_PCT","E_PACE","PACE","PACE_PER40",
            "POSS","PIE",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert pandas NA/NaT to None
        df = df.where(pd.notna(df), None)

        print(f"Loading {len(df)} rows into database...")

        columns = [col for col in df.columns if col not in columns_to_remove]
        column_names = ", ".join([col.lower() for col in columns])
        placeholders = ", ".join(["%s"] * len(columns))

        insert_query = sql.SQL(
            """
            INSERT INTO {}.{} ({})
            VALUES ({})
            ON CONFLICT (game_id, team_id, player_id, season_year) DO NOTHING
            """
        ).format(
            sql.Identifier(schema),
            sql.Identifier(schema),
            sql.SQL(column_names),
            sql.SQL(placeholders),
        )

        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            values = [tuple(row) for row in batch[columns].values]
            cur.executemany(insert_query, values)
            conn.commit()
            total_inserted += len(batch)
            print(f"Inserted {total_inserted}/{len(df)} rows...")

        print(f"\nSuccessfully loaded {total_inserted} rows into the database!")

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
    if not create_players_schema_database():
        raise SystemExit(1)

    print("\nStep 2: Creating schema + table...")
    if not create_players_table():
        raise SystemExit(1)

    print("\nStep 3: Loading combined data...")
    from combine_games_data import combine_all_nba_games

    data_dir = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    df_all_players = combine_all_nba_games(data_dir, file_prefix="nba_players")

    if df_all_players is None:
        print("Failed to load combined dataframe!")
        raise SystemExit(1)

    print("\nStep 4: Inserting data into PostgreSQL...")
    if not load_players_data_to_db(df_all_players):
        raise SystemExit(1)

    print("\nDatabase setup complete.")