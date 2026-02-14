import pandas as pd
from nba_ou.data_preparation.team.cleaning_teams import fix_home_away_parsing_errors
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_games,
)
from psycopg import sql


def load_games_from_db(seasons=None) -> pd.DataFrame | None:
    schema = get_schema_name_games()
    table = schema  # convention: schema == table

    conn = None
    try:
        conn = connect_nba_db()

        if seasons is not None:
            season_years = [int(s.split("-")[0]) for s in seasons]

            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
                WHERE season_year = ANY(%s)
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn, params=(season_years,))

        else:
            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} game records from database")
        return df

    except Exception as e:
        print(f"Error loading games from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()


def load_cleaned_games_for_odds(
    season_year: str | int | None = None,
) -> pd.DataFrame:
    """
    Load games from DB, filter out All Star and Preseason games, and fix home/away issues.

    Args:
        season_year: Optional season start year (e.g., 2025 for 2025-26 season).
                     If None, loads all seasons.

    Returns:
        Cleaned DataFrame with regular season and playoff games only.
    """
    # Load games from database
    if season_year is not None:
        season_str = f"{season_year}-{str(int(season_year) + 1)[-2:]}"
        df = load_games_from_db(seasons=[season_str])
    else:
        df = load_games_from_db(seasons=None)

    if df is None or df.empty:
        return pd.DataFrame()

    # Filter out All Star and Preseason games
    if "season_type" in df.columns:
        df = df[~df["season_type"].isin(["All Star", "Preseason"])]

    # Fix home/away parsing errors
    df = fix_home_away_parsing_errors(df)

    df = df.drop_duplicates(keep ="first").reset_index(drop=True)

    return df
