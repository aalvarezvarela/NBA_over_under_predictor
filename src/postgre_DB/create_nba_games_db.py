import os

import pandas as pd
import psycopg

# Database configuration
DB_NAME = "nba_games"
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
    """Create the nba_games table with appropriate data types."""
    try:
        conn = connect_app_db()
        cursor = conn.cursor()

        # Drop table if exists (for fresh start)
        cursor.execute("DROP TABLE IF EXISTS nba_games CASCADE")

        # Create table with composite primary key
        create_table_query = """
        CREATE TABLE nba_games (
            SEASON_ID TEXT NOT NULL,
            TEAM_ID TEXT NOT NULL,
            TEAM_ABBREVIATION VARCHAR(10),
            TEAM_NAME VARCHAR(100),
            GAME_ID TEXT NOT NULL,
            GAME_DATE DATE NOT NULL,
            MATCHUP VARCHAR(50),
            WL VARCHAR(1),
            MIN INTEGER,
            PTS INTEGER NOT NULL,
            FGM INTEGER,
            FGA INTEGER,
            FG_PCT NUMERIC(5, 3),
            FG3M INTEGER,
            FG3A INTEGER,
            FG3_PCT NUMERIC(5, 3),
            FTM INTEGER,
            FTA INTEGER,
            FT_PCT NUMERIC(5, 3),
            OREB INTEGER,
            DREB INTEGER,
            REB INTEGER,
            AST INTEGER,
            STL INTEGER,
            BLK INTEGER,
            TOV INTEGER,
            PF INTEGER,
            PLUS_MINUS INTEGER,
            SEASON_TYPE VARCHAR(20),
            HOME BOOLEAN,
            TEAM_CITY VARCHAR(50),
            E_OFF_RATING NUMERIC(8, 3),
            OFF_RATING NUMERIC(8, 3),
            E_DEF_RATING NUMERIC(8, 3),
            DEF_RATING NUMERIC(8, 3),
            E_NET_RATING NUMERIC(8, 3),
            NET_RATING NUMERIC(8, 3),
            AST_PCT NUMERIC(8, 3),
            AST_TOV NUMERIC(8, 3),
            AST_RATIO NUMERIC(8, 3),
            OREB_PCT NUMERIC(8, 3),
            DREB_PCT NUMERIC(8, 3),
            REB_PCT NUMERIC(8, 3),
            E_TM_TOV_PCT NUMERIC(8, 3),
            TM_TOV_PCT NUMERIC(8, 3),
            EFG_PCT NUMERIC(8, 3),
            TS_PCT NUMERIC(8, 3),
            USG_PCT NUMERIC(8, 3),
            E_USG_PCT NUMERIC(8, 3),
            E_PACE NUMERIC(8, 3),
            PACE NUMERIC(8, 3),
            PACE_PER40 NUMERIC(8, 3),
            POSS NUMERIC(8, 3),
            PIE NUMERIC(8, 3),
            PRIMARY KEY (GAME_ID, TEAM_ID, SEASON_ID)
        )
        """

        cursor.execute(create_table_query)
        conn.commit()
        print("Table 'nba_games' created successfully with composite primary key!")

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX idx_game_date ON nba_games(GAME_DATE)")
        cursor.execute("CREATE INDEX idx_team_id ON nba_games(TEAM_ID)")
        cursor.execute("CREATE INDEX idx_season_id ON nba_games(SEASON_ID)")
        conn.commit()
        print("Indexes created successfully!")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating table: {e}")
        return False


def load_data_to_db(df):
    """Load the combined dataframe into PostgreSQL."""
    try:
        # Remove teamSlug column if it exists
        if "teamSlug" in df.columns:
            df = df.drop(columns=["teamSlug"])
            print("Removed 'teamSlug' column")

        conn = connect_app_db()
        cursor = conn.cursor()

        # Convert data types
        print("Converting data types...")

        # Handle WL column - replace empty/NaN with None (NULL)
        if "WL" in df.columns:
            df["WL"] = df["WL"].replace("", None)
            df["WL"] = df["WL"].where(df["WL"].notna(), None)

        # Convert HOME to boolean
        if "HOME" in df.columns:
            df["HOME"] = df["HOME"].map(
                {"True": True, "False": False, True: True, False: False}
            )

        # Convert GAME_DATE to date format
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

        # Convert numeric columns (those that aren't IDs or text)
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

        # Convert all pandas NA/NaT values to None (NULL) for PostgreSQL compatibility
        print("Converting pandas NA values to None...")
        df = df.where(pd.notna(df), None)

        print(f"Loading {len(df)} rows into database...")

        # Prepare column names (excluding teamSlug)
        columns = [col for col in df.columns if col != "teamSlug"]
        column_names = ", ".join([f'"{col}"' for col in columns])  # Quote column names
        placeholders = ", ".join(["%s"] * len(columns))

        insert_query = f"""
        INSERT INTO nba_games ({column_names})
        VALUES ({placeholders})
        ON CONFLICT ("GAME_ID", "TEAM_ID", "SEASON_ID") DO NOTHING
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
        cursor.execute("SELECT COUNT(*) FROM nba_games")
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

    print("\nStep 3: Loading combined data...")
    # Load the combined dataframe
    from combine_games_data import combine_all_nba_games

    data_dir = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    )
    df_all_games = combine_all_nba_games(data_dir)

    if df_all_games is not None:
        print("\nStep 4: Inserting data into PostgreSQL...")
        load_data_to_db(df_all_games)
    else:
        print("Failed to load combined dataframe!")
        exit(1)

    print("\n✅ Database setup complete!")
