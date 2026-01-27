import os

import pandas as pd
import psycopg
from nba_ou.fetch_data.fetch_odds_data.odds_mgm.get_odds_mgm import (
    build_mgm_odds_df_from_events,
)
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    connect_postgres_db,
    get_db_credentials,
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
                    game_id SERIAL NOT NULL,
                    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    game_date_local TIMESTAMP WITHOUT TIME ZONE NOT NULL,
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
                    mgm_total_under_money NUMERIC(8, 4),
                    PRIMARY KEY (game_id)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_odds_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_home)").format(
                    sql.Identifier("idx_odds_team_home"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_away)").format(
                    sql.Identifier("idx_odds_team_away"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.nba_odds' created successfully!")
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


def load_odds_mgm_data_to_db(conn: psycopg.Connection | None = None) -> bool:
    """Load odds data from CSV into schema SCHEMA_NAME_ODDS_MGM table nba_odds_mgm."""
    close_conn = False
    schema = get_schema_name_odds_mgm()


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


def create_odds_df_from_jsons(path: str) -> pd.DataFrame:
    """Create the odds mgm database and table."""
    
    
    events = build_mgm_odds_df_from_events(path)
    


if __name__ == "__main__":
    print("\nStep 2: Creating schema + nba_odds table...")
    if not create_odds_mgm_table():
        raise SystemExit(1)

    print("\nStep 3: Loading odds data...")
    odds_csv_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/odds_data/odds_data.csv"

    if load_odds_mgm_data_to_db(odds_csv_path):
        print("\nOdds setup complete.")

        print("\nSample of recent odds:")
        recent_odds = get_recent_odds(5)
        for odds in recent_odds:
            print(f"\n{odds['game_date']}: {odds['team_home']} vs {odds['team_away']}")
            print(f"  Total Line (avg): {odds['average_total_line']}")
            print(
                f"  Spread (avg): Home {odds['average_spread_home']}, Away {odds['average_spread_away']}"
            )
    else:
        print("Failed to load odds data!")
        raise SystemExit(1)
