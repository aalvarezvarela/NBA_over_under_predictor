import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_players,
)
from psycopg import sql


def load_players_from_db(seasons=None) -> pd.DataFrame | None:
    """
    Load NBA players data from Postgres (single DB, schema.table).

    Args:
        seasons: list like ['2023-24', '2024-25'] or None for all seasons.

    Returns:
        DataFrame or None if failure.
    """
    schema = get_schema_name_players()
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
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn, params=(season_years,))
        else:
            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} player records from database")
        return df

    except Exception as e:
        print(f"Error loading players from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()
