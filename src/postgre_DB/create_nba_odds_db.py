import os

import pandas as pd
import psycopg
from psycopg import sql

from .db_config import (
    connect_nba_db,
    get_schema_name_odds,
)


def create_odds_schema_if_not_exists(conn: psycopg.Connection, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_odds_table(drop_existing: bool = True):
    """Create the nba_odds table inside schema SCHEMA_NAME_ODDS."""
    try:
        schema = get_schema_name_odds()
        conn = connect_nba_db()

        create_odds_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier("nba_odds"),
                    )
                )

            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    team_home VARCHAR(100) NOT NULL,
                    team_away VARCHAR(100) NOT NULL,
                    most_common_total_line NUMERIC(8, 4),
                    average_total_line NUMERIC(8, 4),
                    most_common_moneyline_home NUMERIC(10, 4),
                    average_moneyline_home NUMERIC(10, 4),
                    most_common_moneyline_away NUMERIC(10, 4),
                    average_moneyline_away NUMERIC(10, 4),
                    most_common_spread_home NUMERIC(8, 4),
                    average_spread_home NUMERIC(8, 4),
                    most_common_spread_away NUMERIC(8, 4),
                    average_spread_away NUMERIC(8, 4),
                    average_total_over_money NUMERIC(8, 4),
                    average_total_under_money NUMERIC(8, 4),
                    most_common_total_over_money NUMERIC(8, 4),
                    most_common_total_under_money NUMERIC(8, 4),
                    PRIMARY KEY (game_date, team_home, team_away)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier("nba_odds"))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_odds_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier("nba_odds"),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_home)").format(
                    sql.Identifier("idx_odds_team_home"),
                    sql.Identifier(schema),
                    sql.Identifier("nba_odds"),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_away)").format(
                    sql.Identifier("idx_odds_team_away"),
                    sql.Identifier(schema),
                    sql.Identifier("nba_odds"),
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


def load_odds_data_to_db(csv_path: str, conn: psycopg.Connection | None = None) -> bool:
    """Load odds data from CSV into schema SCHEMA_NAME_ODDS table nba_odds."""
    close_conn = False
    schema = get_schema_name_odds()

    try:
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return False

        print(f"Reading odds data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")

        df = coalesce_duplicate_columns(df)

        if conn is None:
            conn = connect_nba_db()
            close_conn = True

        create_odds_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            print("Converting data types...")

            # Ensure required columns exist (in case some are missing)
            for new_col in [
                "average_total_over_money",
                "average_total_under_money",
                "most_common_total_over_money",
                "most_common_total_under_money",
            ]:
                if new_col not in df.columns:
                    df[new_col] = None

            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(
                    df["game_date"], errors="coerce", utc=True
                )

            numeric_cols = [
                "most_common_total_line",
                "average_total_line",
                "most_common_moneyline_home",
                "average_moneyline_home",
                "most_common_moneyline_away",
                "average_moneyline_away",
                "most_common_spread_home",
                "average_spread_home",
                "most_common_spread_away",
                "average_spread_away",
                "average_total_over_money",
                "average_total_under_money",
                "most_common_total_over_money",
                "most_common_total_under_money",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.where(pd.notna(df), None)

            print(f"Loading {len(df)} rows into database...")

            columns = df.columns.tolist()
            column_names = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))

            insert_query = sql.SQL(
                f"""
                INSERT INTO {{}}.{{}} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT (game_date, team_home, team_away) DO UPDATE SET
                    most_common_total_line = EXCLUDED.most_common_total_line,
                    average_total_line = EXCLUDED.average_total_line,
                    most_common_moneyline_home = EXCLUDED.most_common_moneyline_home,
                    average_moneyline_home = EXCLUDED.average_moneyline_home,
                    most_common_moneyline_away = EXCLUDED.most_common_moneyline_away,
                    average_moneyline_away = EXCLUDED.average_moneyline_away,
                    most_common_spread_home = EXCLUDED.most_common_spread_home,
                    average_spread_home = EXCLUDED.average_spread_home,
                    most_common_spread_away = EXCLUDED.most_common_spread_away,
                    average_spread_away = EXCLUDED.average_spread_away,
                    average_total_over_money = EXCLUDED.average_total_over_money,
                    average_total_under_money = EXCLUDED.average_total_under_money,
                    most_common_total_over_money = EXCLUDED.most_common_total_over_money,
                    most_common_total_under_money = EXCLUDED.most_common_total_under_money
                """
            ).format(sql.Identifier(schema), sql.Identifier("nba_odds"))

            batch_size = 1000
            total_processed = 0
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                values = [tuple(row) for row in batch[columns].values]
                cur.executemany(insert_query, values)
                conn.commit()
                total_processed += len(batch)
                print(f"Processed {total_processed}/{len(df)} rows...")

            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier("nba_odds")
                )
            )
            count = cur.fetchone()[0]
            print(f"Total rows in {schema}.nba_odds: {count}")

        if close_conn:
            conn.close()
        return True

    except Exception as e:
        print(f"Error loading odds data: {e}")
        import traceback

        traceback.print_exc()
        if close_conn and conn:
            conn.close()
        return False


def get_odds_for_game(game_date, team_home, team_away):
    """Retrieve odds data for a specific game from schema SCHEMA_NAME_ODDS."""
    schema = get_schema_name_odds()
    try:
        conn = connect_nba_db()
        with conn.cursor() as cur:
            query = sql.SQL(
                "SELECT * FROM {}.{} WHERE game_date = %s AND team_home = %s AND team_away = %s"
            ).format(sql.Identifier(schema), sql.Identifier("nba_odds"))

            cur.execute(query, (game_date, team_home, team_away))
            row = cur.fetchone()

            if row:
                columns = [desc[0] for desc in cur.description]
                result = dict(zip(columns, row))
            else:
                result = None

        conn.close()
        return result
    except Exception as e:
        print(f"Error retrieving odds data: {e}")
        return None


def get_recent_odds(limit: int = 10):
    """Retrieve the most recent odds data from schema SCHEMA_NAME_ODDS."""
    schema = get_schema_name_odds()
    try:
        conn = connect_nba_db()
        with conn.cursor() as cur:
            query = sql.SQL(
                "SELECT * FROM {}.{} ORDER BY game_date DESC LIMIT %s"
            ).format(sql.Identifier(schema), sql.Identifier("nba_odds"))

            cur.execute(query, (limit,))
            rows = cur.fetchall()

            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in rows]

        conn.close()
        return results
    except Exception as e:
        print(f"Error retrieving recent odds: {e}")
        return []


if __name__ == "__main__":


    print("\nStep 2: Creating schema + nba_odds table...")
    if not create_odds_table():
        raise SystemExit(1)

    print("\nStep 3: Loading odds data...")
    odds_csv_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/odds_data/odds_data.csv"

    if load_odds_data_to_db(odds_csv_path):
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
