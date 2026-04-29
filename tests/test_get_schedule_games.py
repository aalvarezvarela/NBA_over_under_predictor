from importlib import import_module

import pandas as pd
from nba_ou.fetch_data.scheduled_game.get_schedule_games import (
    EASTERN_TZ,
    _parse_scoreboard_game_times,
    filter_started_games,
)


def test_parse_scoreboard_game_times_keeps_tbd_as_unknown_time() -> None:
    games = pd.DataFrame(
        [
            {
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
            }
        ]
    )

    result = _parse_scoreboard_game_times(games)

    assert pd.isna(result.iloc[0])


def test_parse_scoreboard_game_times_extracts_tipoff_time() -> None:
    games = pd.DataFrame(
        [
            {
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "7:30 PM ET",
            }
        ]
    )

    result = _parse_scoreboard_game_times(games)

    assert result.iloc[0] == pd.Timestamp("2026-04-29 19:30", tz=EASTERN_TZ)


def test_filter_started_games_retains_unknown_times_and_future_known_times() -> None:
    games = pd.DataFrame(
        [
            {"GAME_ID": "unknown", "GAME_TIME": pd.NaT},
            {
                "GAME_ID": "started",
                "GAME_TIME": pd.Timestamp("2026-04-29 18:00", tz=EASTERN_TZ),
            },
            {
                "GAME_ID": "future",
                "GAME_TIME": pd.Timestamp("2026-04-29 21:00", tz=EASTERN_TZ),
            },
        ]
    )

    result = filter_started_games(
        games,
        now_et=pd.Timestamp("2026-04-29 19:00", tz=EASTERN_TZ),
    )

    assert result["GAME_ID"].tolist() == ["unknown", "future"]


def test_get_schedule_games_drops_invalid_status_without_dropping_tbd(
    monkeypatch,
) -> None:
    schedule_module = import_module("nba_ou.fetch_data.scheduled_game.get_schedule_games")
    games = pd.DataFrame(
        [
            {
                "GAME_ID": "tbd",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
                "HOME_TEAM_ID": "1610612738",
                "VISITOR_TEAM_ID": "1610612752",
            },
            {
                "GAME_ID": "invalid",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "Final",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            },
        ]
    )
    monkeypatch.setattr(schedule_module, "nba_api_schedule_games", lambda _: games)
    monkeypatch.setattr(schedule_module, "filter_started_games", lambda games: games)

    result = schedule_module.get_schedule_games("2026-04-29")

    assert result["GAME_ID"].tolist() == ["tbd"]
    assert pd.isna(result["GAME_TIME"].iloc[0])


def test_get_schedule_games_keeps_tbd_and_future_timed_games(monkeypatch) -> None:
    schedule_module = import_module("nba_ou.fetch_data.scheduled_game.get_schedule_games")
    games = pd.DataFrame(
        [
            {
                "GAME_ID": "tbd",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
                "HOME_TEAM_ID": "1610612738",
                "VISITOR_TEAM_ID": "1610612752",
            },
            {
                "GAME_ID": "timed",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "9:30 PM ET",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            },
        ]
    )
    monkeypatch.setattr(schedule_module, "nba_api_schedule_games", lambda _: games)
    filter_started_games_original = schedule_module.filter_started_games
    monkeypatch.setattr(
        schedule_module,
        "filter_started_games",
        lambda games: filter_started_games_original(
            games,
            now_et=pd.Timestamp("2026-04-29 19:00", tz=EASTERN_TZ),
        ),
    )

    result = schedule_module.get_schedule_games("2026-04-29")

    assert result["GAME_ID"].tolist() == ["tbd", "timed"]
    assert pd.isna(result.loc[result["GAME_ID"] == "tbd", "GAME_TIME"].iloc[0])
    assert result.loc[result["GAME_ID"] == "timed", "GAME_TIME"].iloc[0] == pd.Timestamp(
        "2026-04-29 21:30",
        tz=EASTERN_TZ,
    )
