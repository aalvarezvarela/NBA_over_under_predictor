import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_injuries,
)
from psycopg import sql


def load_injury_data_from_db(seasons):
    """
    Load injury data from database for the specified seasons.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])

    Returns:
        pd.DataFrame: Combined injury data for all seasons
    """

    schema = get_schema_name_injuries()
    table = schema

    conn = None
    try:
        conn = connect_nba_db()

        # Convert season format from "2023-24" to 2023 (year only)
        season_years = [int(s.split("-")[0]) for s in seasons]

        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            WHERE season_year = ANY(%s)
        """).format(sql.Identifier(schema), sql.Identifier(table))

        query = query_obj.as_string(conn)
        df_injuries = pd.read_sql_query(query, conn, params=(season_years,))

        # Convert column names to uppercase to match expected format
        df_injuries.columns = df_injuries.columns.str.upper()

        # Ensure TEAM_ID and PLAYER_ID are strings
        if "TEAM_ID" in df_injuries.columns:
            df_injuries["TEAM_ID"] = df_injuries["TEAM_ID"].astype(str)
        if "PLAYER_ID" in df_injuries.columns:
            df_injuries["PLAYER_ID"] = df_injuries["PLAYER_ID"].astype(str)

        # Remove duplicates
        df_injuries = df_injuries.drop_duplicates()

        print(f"Loaded {len(df_injuries)} injury records from database")
        return df_injuries

    except Exception as e:
        print(f"Error loading injury data from database: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    finally:
        if conn is not None:
            conn.close()
