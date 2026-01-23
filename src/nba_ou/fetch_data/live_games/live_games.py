from nba_api.live.nba.endpoints import scoreboard


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
