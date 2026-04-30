import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.odds_sportsbook_line_history.create_db.create_odds_sportsbook_line_history_db import (
    get_line_history_schema_name,
)
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (
    normalize_line_history_market,
)
from psycopg import sql


def _normalize_game_ids(game_ids) -> list[str]:
    if game_ids is None:
        return []

    normalized = []
    seen = set()
    for game_id in game_ids:
        if pd.isna(game_id):
            continue
        game_id_str = str(game_id).strip()
        if not game_id_str or game_id_str in seen:
            continue
        seen.add(game_id_str)
        normalized.append(game_id_str)
    return normalized


def load_odds_sportsbook_line_history_from_db(
    market: str,
    seasons=None,
    extra_game_ids=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame | None:
    normalized_market = normalize_line_history_market(market)
    schema = get_line_history_schema_name(normalized_market)
    table = schema

    conn = None
    try:
        conn = connect_nba_db()
        where_parts = []
        query_params = []

        if seasons is not None and len(seasons) > 0:
            season_years = [int(str(s).split("-")[0]) for s in seasons]
            where_parts.append(sql.SQL("season_year = ANY(%s)"))
            query_params.append(season_years)

        normalized_extra_game_ids = _normalize_game_ids(extra_game_ids)
        if normalized_extra_game_ids:
            where_parts.append(sql.SQL("game_id = ANY(%s)"))
            query_params.append(normalized_extra_game_ids)

        if start_date is not None:
            where_parts.append(sql.SQL("game_date >= %s"))
            query_params.append(pd.to_datetime(start_date).date())

        if end_date is not None:
            where_parts.append(sql.SQL("game_date <= %s"))
            query_params.append(pd.to_datetime(end_date).date())

        where_clause = sql.SQL("")
        if where_parts:
            where_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts)

        query_obj = sql.SQL(
            """
            SELECT *
            FROM {}.{}
            {}
            ORDER BY game_date DESC, game_id DESC, bookmaker_slug, line_timestamp
        """
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            where_clause,
        )

        query = query_obj.as_string(conn)
        if query_params:
            df = pd.read_sql_query(query, conn, params=tuple(query_params))
        else:
            df = pd.read_sql_query(query, conn)

        print(
            f"Loaded {len(df)} {normalized_market} sportsbook line-history records from database"
        )
        return df

    except Exception as e:
        print(f"Error loading {normalized_market} sportsbook line history: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()
