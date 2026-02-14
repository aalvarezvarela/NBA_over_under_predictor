import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_refs
from psycopg import sql


def get_refs_data_from_db(seasons):
    """
    Load referee data from database for the specified seasons.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])

    Returns:
        pd.DataFrame: Combined referee data for all seasons
    """

    schema = get_schema_name_refs()
    table = "nba_refs"  # table name in refs schema

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
        df_refs = pd.read_sql_query(query, conn, params=(season_years,))

        # Convert column names to uppercase to match expected format
        df_refs.columns = df_refs.columns.str.upper()

        # Ensure GAME_ID is string for consistent merging
        if "GAME_ID" in df_refs.columns:
            df_refs["GAME_ID"] = df_refs["GAME_ID"].astype(str)

        # Remove duplicates
        df_refs = df_refs.drop_duplicates()

        print(f"Loaded {len(df_refs)} referee records from database")
        return df_refs

    except Exception as e:
        print(f"Error loading referee data from database: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    finally:
        if conn is not None:
            conn.close()
