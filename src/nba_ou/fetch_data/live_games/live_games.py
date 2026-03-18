from datetime import datetime

from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import ScoreboardV2


def get_live_game_ids():
    """
    Return a list of GAME_IDs for NBA games currently in progress.
    """
    sb = scoreboard.ScoreBoard()  # defaults to today's games
    games = sb.games.get_dict()  # list[dict]
    live_games = [
        g for g in games if g.get("gameStatus") == 2
    ]  # 1=scheduled, 2=live, 3=final (typical)
    return [g["gameId"] for g in live_games]


def get_game_ids_for_date(game_date: str) -> list[str]:
    """
    Return all NBA GAME_IDs scheduled on a given date.
    """
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.") from exc

    games = ScoreboardV2(game_date=game_date).get_data_frames()[0]
    if games.empty or "GAME_ID" not in games.columns:
        return []

    return games["GAME_ID"].dropna().astype(str).unique().tolist()
