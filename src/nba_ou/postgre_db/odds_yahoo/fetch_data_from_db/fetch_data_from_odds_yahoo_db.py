import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_yahoo,
)
from psycopg import sql


def load_odds_yahoo_from_db(seasons=None) -> pd.DataFrame | None:
    schema = get_schema_name_odds_yahoo()
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

        print(f"Loaded {len(df)} Yahoo odds records from database")
        return df

    except Exception as e:
        print(f"Error loading Yahoo odds from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()
