# from utils.general_utils import get_nba_season_nullable
import configparser
import os

import numpy as np
import pandas as pd
import psycopg
from nba_api.stats.endpoints import LeagueGameFinder
from utils.general_utils import get_nba_season_nullable

from .db_config import connect_predictions_db


def get_predictions_table_name():
    config = configparser.ConfigParser()
    config.read(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.ini"
        )
    )
    return config.get("Database", "DB_NAME_PREDICTIONS", fallback="nba_predictions")


def get_game_ids_with_null_total_scored_points() -> pd.DataFrame:
    table_name = get_predictions_table_name()
    conn = connect_predictions_db()
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE total_scored_points IS NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    game_ids = df["game_id"].dropna().unique().tolist()
    if not game_ids:
        return pd.DataFrame(
            columns=["game_id", "total_scored_points"]
        )  # Return empty DataFrame if no game_ids
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

    # return only what you need to update
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

    table_name = get_predictions_table_name()
    sql = f"""
        UPDATE {table_name}
        SET total_scored_points = %s
        WHERE game_id = %s
    """

    with connect_predictions_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
