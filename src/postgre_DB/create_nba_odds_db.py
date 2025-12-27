import os

import pandas as pd
import psycopg

from .db_config import (
    connect_odds_db as connect_app_db,
)

# Import database configuration from centralized config
from .db_config import (
    connect_postgres_db,
    get_odds_db_name,
)


def create_database():
    """Create the PostgreSQL database if it doesn't exist."""
    try:
        db_name = get_odds_db_name()
        # Connect to PostgreSQL server
        conn = connect_postgres_db()
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,)
        )
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully!")
        else:
            print(f"Database '{db_name}' already exists.")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def create_odds_table():
    """Create the nba_odds table with appropriate data types."""
    try:
        conn = connect_app_db()
        cursor = conn.cursor()

        # Drop table if exists (for fresh start)
        cursor.execute("DROP TABLE IF EXISTS nba_odds CASCADE")

        # Create table with composite primary key
        create_table_query = """
        CREATE TABLE nba_odds (
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
            PRIMARY KEY (game_date, team_home, team_away)
        )
        """

        cursor.execute(create_table_query)
        conn.commit()
        print("Table 'nba_odds' created successfully with composite primary key!")

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX idx_odds_game_date ON nba_odds(game_date)")
        cursor.execute("CREATE INDEX idx_odds_team_home ON nba_odds(team_home)")
        cursor.execute("CREATE INDEX idx_odds_team_away ON nba_odds(team_away)")
        conn.commit()
        print("Indexes created successfully!")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating odds table: {e}")
        return False


def load_odds_data_to_db(csv_path, conn=None):
    """Load odds data from CSV into PostgreSQL.

    Args:
        csv_path: Path to the odds CSV file
        conn: Optional database connection. If None, creates a new connection.

    Returns:
        bool: True if successful, False otherwise
    """
    close_conn = False
    try:
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return False

        # Read the CSV file
        print(f"Reading odds data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")

        if conn is None:
            conn = connect_app_db()
            close_conn = True
        cursor = conn.cursor()

        # Convert data types
        print("Converting data types...")

        # Convert game_date to timestamp
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

        # Convert numeric columns
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
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert all pandas NA/NaT values to None (NULL) for PostgreSQL compatibility
        print("Converting pandas NA values to None...")
        df = df.where(pd.notna(df), None)

        print(f"Loading {len(df)} rows into database...")

        # Prepare column names
        columns = df.columns.tolist()
        column_names = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))

        insert_query = f"""
        INSERT INTO nba_odds ({column_names})
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
            average_spread_away = EXCLUDED.average_spread_away
        """

        # Insert data in batches
        batch_size = 1000
        total_inserted = 0

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            values = [tuple(row) for row in batch[columns].values]
            cursor.executemany(insert_query, values)
            conn.commit()
            total_inserted += len(batch)
            print(f"Inserted {total_inserted}/{len(df)} rows...")

        print(f"\nSuccessfully loaded {total_inserted} rows into the database!")

        # Verify count
        cursor.execute("SELECT COUNT(*) FROM nba_odds")
        count = cursor.fetchone()[0]
        print(f"Total rows in nba_odds table: {count}")

        cursor.close()
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
    """Retrieve odds data for a specific game.

    Args:
        game_date: Game date (datetime object or string)
        team_home: Home team name
        team_away: Away team name

    Returns:
        dict: Odds data for the game, or None if not found
    """
    try:
        conn = connect_app_db()
        cursor = conn.cursor()

        query = """
        SELECT * FROM nba_odds
        WHERE game_date = %s AND team_home = %s AND team_away = %s
        """

        cursor.execute(query, (game_date, team_home, team_away))
        row = cursor.fetchone()

        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
        else:
            result = None

        cursor.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Error retrieving odds data: {e}")
        return None


def get_recent_odds(limit=10):
    """Retrieve the most recent odds data.

    Args:
        limit: Number of records to retrieve

    Returns:
        list: List of dictionaries containing odds data
    """
    try:
        conn = connect_app_db()
        cursor = conn.cursor()

        query = """
        SELECT * FROM nba_odds
        ORDER BY game_date DESC
        LIMIT %s
        """

        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error retrieving recent odds: {e}")
        return []


if __name__ == "__main__":
    print("Step 1: Creating database...")
    if not create_database():
        exit(1)

    print("\nStep 2: Creating nba_odds table...")
    if not create_odds_table():
        exit(1)

    print("\nStep 3: Loading odds data...")
    odds_csv_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/odds_data/odds_data.csv"

    if load_odds_data_to_db(odds_csv_path):
        print("\n✅ Odds database setup complete!")

        # Display some sample data
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
        exit(1)
