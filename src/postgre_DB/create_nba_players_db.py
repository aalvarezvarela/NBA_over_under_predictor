import os

import pandas as pd
import psycopg

# Database configuration
DB_NAME = "nba_players"
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
    """Create the nba_players table with appropriate data types."""
    try:
        conn = connect_app_db()
        cursor = conn.cursor()

        # Drop table if exists (for fresh start)
        cursor.execute("DROP TABLE IF EXISTS nba_players CASCADE")

        # Create table with composite primary key
        create_table_query = """
        CREATE TABLE nba_players (
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            team_abbreviation VARCHAR(10),
            team_city VARCHAR(50),
            team_name VARCHAR(100),
            player_id TEXT NOT NULL,
            player_name VARCHAR(100),
            nickname VARCHAR(50),
            firstname VARCHAR(50),
            familyname VARCHAR(50),
            start_position VARCHAR(10),
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
            PRIMARY KEY (game_id, team_id, player_id)
        )
        """

        cursor.execute(create_table_query)
        conn.commit()
        print("Table 'nba_players' created successfully with composite primary key!")

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX idx_player_game_id ON nba_players(game_id)")
        cursor.execute("CREATE INDEX idx_player_team_id ON nba_players(team_id)")
        cursor.execute("CREATE INDEX idx_player_id ON nba_players(player_id)")
        cursor.execute("CREATE INDEX idx_player_name ON nba_players(player_name)")
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
        # Remove unnecessary slug columns
        columns_to_remove = ["teamSlug", "playerSlug"]
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Removed '{col}' column")

        conn = connect_app_db()
        cursor = conn.cursor()

        # Convert data types
        print("Converting data types...")

        # Convert numeric columns (those that aren't IDs or text)
        integer_cols = [
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
            "PTS",
        ]

        for col in integer_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                )
                # Replace pandas NA with None for psycopg compatibility
                df[col] = df[col].astype(object).where(df[col].notna(), None)

        float_cols = [
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT",
            "PLUS_MINUS",
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
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert all pandas NA/NaT values to None (NULL) for PostgreSQL compatibility
        print("Converting pandas NA values to None...")
        df = df.where(pd.notna(df), None)

        print(f"Loading {len(df)} rows into database...")

        # Prepare column names (excluding removed columns) - lowercase to match table schema
        columns = [col for col in df.columns if col not in columns_to_remove]
        column_names = ", ".join([col.lower() for col in columns])
        placeholders = ", ".join(["%s"] * len(columns))

        insert_query = f"""
        INSERT INTO nba_players ({column_names})
        VALUES ({placeholders})
        ON CONFLICT (game_id, team_id, player_id) DO NOTHING
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
        cursor.execute("SELECT COUNT(*) FROM nba_players")
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
    df_all_players = combine_all_nba_games(data_dir, file_prefix="nba_players")

    if df_all_players is not None:
        print("\nStep 4: Inserting data into PostgreSQL...")
        load_data_to_db(df_all_players)
    else:
        print("Failed to load combined dataframe!")
        exit(1)

    print("\n✅ Database setup complete!")
