import pandas as pd
import psycopg

# Database configuration
DB_USER = "adrian_alvarez"
DB_PASSWORD = "12345"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"


def connect_to_db(db_name):
    """Connect to a specific PostgreSQL database."""
    return psycopg.connect(
        dbname=db_name,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def load_games_from_db(seasons=None):
    """
    Load NBA games data from PostgreSQL database.

    Args:
        seasons (list, optional): List of season strings to filter (e.g., ['2023-24', '2024-25']).
                                 If None, loads all seasons.

    Returns:
        pd.DataFrame: DataFrame with all games data.
    """
    try:
        conn = connect_to_db("nba_games")

        if seasons is not None:
            # Extract the starting year from season format (e.g., '2023-24' -> 2023)
            season_years = [int(s.split("-")[0]) for s in seasons]

            placeholders = ", ".join(["%s"] * len(season_years))
            query = f"""
                SELECT * FROM nba_games 
                WHERE season_year IN ({placeholders})
                ORDER BY game_date DESC
            """
            df = pd.read_sql_query(query, conn, params=season_years)
        else:
            query = "SELECT * FROM nba_games ORDER BY game_date DESC"
            df = pd.read_sql_query(query, conn)

        conn.close()
        print(f"Loaded {len(df)} game records from database")
        return df

    except Exception as e:
        print(f"Error loading games from database: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_players_from_db(seasons=None):
    """
    Load NBA players data from PostgreSQL database.

    Args:
        seasons (list, optional): List of season strings to filter (e.g., ['2023-24', '2024-25']).
                                 If None, loads all seasons.

    Returns:
        pd.DataFrame: DataFrame with all players data.
    """
    try:
        conn = connect_to_db("nba_players")

        if seasons is not None:
            # Extract the starting year from season format (e.g., '2023-24' -> 2023)
            season_years = [int(s.split("-")[0]) for s in seasons]

            placeholders = ", ".join(["%s"] * len(season_years))
            query = f"""
                SELECT * FROM nba_players 
                WHERE season_year IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=season_years)
        else:
            query = "SELECT * FROM nba_players"
            df = pd.read_sql_query(query, conn)

        conn.close()
        print(f"Loaded {len(df)} player records from database")
        return df

    except Exception as e:
        print(f"Error loading players from database: {e}")
        import traceback

        traceback.print_exc()
        return None



def load_all_nba_data_from_db(seasons=None):
    """
    Load all NBA data (games and players) from PostgreSQL databases.
    This function mimics load_all_nba_data() but queries databases instead of CSVs.

    Args:
        seasons (list, optional): List of season strings to filter (e.g., ['2023-24', '2024-25']).
                                 If None, loads all seasons.

    Returns:
        tuple: (df_games, df_players) - DataFrames with games and players data.
    """
    print("Loading NBA data from PostgreSQL databases...")

    df_games = load_games_from_db(seasons=seasons)
    df_players = load_players_from_db(seasons=seasons)

    if df_games is not None and df_players is not None:

        # Convert column names to uppercase to match expected format
        df_players.columns = df_players.columns.str.upper()
        df_games.columns = df_games.columns.str.upper()

        print(
            f"Successfully loaded {len(df_games)} games and {len(df_players)} player records"
        )
        return df_games, df_players
    
    else:
        print("Failed to load data from databases")
        return None, None
