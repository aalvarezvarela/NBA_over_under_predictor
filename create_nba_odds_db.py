import os

import pandas as pd
import psycopg

# Database configuration
DB_NAME = "nba_odds"
DB_USER = "adrian_alvarez"
DB_PASSWORD = "12345"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"


def connect_postgres_db():
    return psycopg.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        autocommit=True,
    )


def connect_app_db():
    return psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def create_database():
    """Create the PostgreSQL database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server
        conn = connect_postgres_db()
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,)
        )
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully!")
        else:
            print(f"Database '{DB_NAME}' already exists.")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def create_table():
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
            most_common_total_line NUMERIC(10, 4),
            average_total_line NUMERIC(10, 4),
            most_common_moneyline_home NUMERIC(10, 4),
            average_moneyline_home NUMERIC(10, 4),
            most_common_moneyline_away NUMERIC(10, 4),
            average_moneyline_away NUMERIC(10, 4),
            most_common_spread_home NUMERIC(10, 4),
            average_spread_home NUMERIC(10, 4),
            most_common_spread_away NUMERIC(10, 4),
            average_spread_away NUMERIC(10, 4),
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
        print(f"Error creating table: {e}")
        return False


def load_data_to_db(csv_path):
    """Load the odds CSV file into PostgreSQL."""
    try:
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

        conn = connect_app_db()
        cursor = conn.cursor()

        # Convert data types
        print("Converting data types...")

        # Convert game_date to datetime
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

        # Convert all numeric columns to float
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
        columns = list(df.columns)
        column_names = ", ".join([col.lower() for col in columns])
        placeholders = ", ".join(["%s"] * len(columns))

        insert_query = f"""
        INSERT INTO nba_odds ({column_names})
        VALUES ({placeholders})
        ON CONFLICT (game_date, team_home, team_away) DO NOTHING
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
        print(f"Total rows in database: {count}")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Step 1: Creating database...")
    if not create_database():
        exit(1)

    print("\nStep 2: Creating table...")
    if not create_table():
        exit(1)

    print("\nStep 3: Loading odds data...")
    # Path to odds CSV file
    odds_csv_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/odds_data/odds_data.csv"

    if os.path.exists(odds_csv_path):
        print("\nStep 4: Inserting data into PostgreSQL...")
        load_data_to_db(odds_csv_path)
    else:
        print(f"Error: File not found: {odds_csv_path}")
        exit(1)

    print("\n✅ Database setup complete!")
