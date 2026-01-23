import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
from psycopg import sql
from nba_ou.utils.general_utils import get_nba_season_nullable

from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_predictions
from nba_ou.postgre_db.predictions.create.create_nba_predictions_db import (
    schema_exists,
)


def get_predictions_schema_and_table() -> tuple[str, str]:
    """
    In your new structure:
      schema = SCHEMA_NAME_PREDICTIONS (e.g. 'nba_predictions')
      table  = same as schema (your convention)
    """
    schema = get_schema_name_predictions()
    table = schema
    return schema, table


def get_game_ids_with_null_total_scored_points() -> pd.DataFrame:
    schema, table = get_predictions_schema_and_table()

    if not schema_exists(schema):
        raise ValueError(f"Schema '{schema}' does not exist in the database.")
    
    conn = connect_nba_db()
    try:
        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            WHERE total_scored_points IS NULL
            OR total_scored_points < %s
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table),
        )

        query = query_obj.as_string(conn)
        df = pd.read_sql_query(query, conn, params=(110,))
    finally:
        conn.close()

    game_ids = df["game_id"].dropna().unique().tolist()
    if not game_ids:
        return pd.DataFrame(columns=["game_id", "total_scored_points"])

    dates = pd.to_datetime(df["game_date"], errors="coerce").dropna().unique()
    seasons = {get_nba_season_nullable(d) for d in dates}

    games = []
    for season in seasons:
        game_finder = LeagueGameFinder(season_nullable=season, league_id_nullable="00")
        g = game_finder.get_data_frames()[0].copy()
        g = g.rename(columns={"GAME_ID": "game_id"})
        games.append(g)

    games = pd.concat(games, ignore_index=True)
    games = games[games["game_id"].isin(game_ids)]

    assert all(games.groupby("game_id").size() == 2), "Not all games have two rows"

    games["total_scored_points"] = games.groupby("game_id")["PTS"].transform("sum")
    games = games[["game_id", "total_scored_points"]].drop_duplicates()

    # NEW: drop games with total score < 140
    games = games[games["total_scored_points"] >= 140]

    updates = games.dropna(subset=["game_id", "total_scored_points"]).copy()
    return updates

def update_total_scored_points(updates: pd.DataFrame) -> None:
    """
    updates must have columns: ['game_id', 'total_scored_points']
    """
    if updates.empty:
        return

    updates = updates.copy()
    updates["game_id"] = updates["game_id"].astype(str)
    updates["total_scored_points"] = pd.to_numeric(
        updates["total_scored_points"], errors="coerce"
    )

    rows = [
        (r["total_scored_points"], r["game_id"])
        for _, r in updates.dropna(subset=["game_id", "total_scored_points"]).iterrows()
    ]
    if not rows:
        return

    schema, table = get_predictions_schema_and_table()

    update_stmt = sql.SQL("""
        UPDATE {}.{}
        SET total_scored_points = %s
        WHERE game_id = %s
    """).format(sql.Identifier(schema), sql.Identifier(table))

    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(update_stmt, rows)
        conn.commit()
    finally:
        conn.close()

def update_total_points_predictions():
    updates = get_game_ids_with_null_total_scored_points()
    print(f"Found {len(updates)} games to update with total scored points.")
    if not updates.empty:
        update_total_scored_points(updates)
        print("Total scored points updated successfully.")
    else:
        print("No updates needed.")
