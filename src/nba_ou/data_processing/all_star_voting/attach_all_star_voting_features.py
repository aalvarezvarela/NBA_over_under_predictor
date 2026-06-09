from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.data_processing.past_injuries.past_injuries import create_player_lookup

TEAM_NAME_BY_ID = {str(v): k for k, v in TEAM_ID_MAP.items()}

FEATURE_COLUMNS = [
    "ALL_STAR_FAN_VOTE_SHARE_BEFORE",
    "ALL_STAR_MIN_SCORE_BEFORE",
    "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE",
    "ALL_STAR_MIN_INJURED_SCORE_BEFORE",
    "ALL_STAR_FAN_VOTES_BEFORE",
    "ALL_STAR_CANDIDATE_COUNT_BEFORE",
    "ALL_STAR_SEASON_YEAR_BEFORE",
]


def all_star_season_year_for_game_date(game_date) -> int:
    d = pd.Timestamp(game_date)
    all_star_calendar_year = d.year if d >= pd.Timestamp(d.year, 3, 1) else d.year - 1
    return all_star_calendar_year - 1


def build_last_team_before_date_lookup(
    df_players: pd.DataFrame,
) -> Callable[[str, pd.Timestamp], str | None]:
    """Returns f(player_id, date) -> last_team_id (str) or None.
    Uses ALL rows in df_players (no MIN > 0 filter - DNPs still indicate team membership).
    """
    required_cols = {"PLAYER_ID", "TEAM_ID", "GAME_DATE"}
    missing_cols = sorted(required_cols - set(df_players.columns))
    if missing_cols:
        raise ValueError(f"df_players is missing columns: {missing_cols}")

    players = df_players[["PLAYER_ID", "TEAM_ID", "GAME_DATE"]].copy()
    players["PLAYER_ID"] = players["PLAYER_ID"].astype(str)
    players["TEAM_ID"] = players["TEAM_ID"].astype(str)
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"], errors="coerce")
    players = players.dropna(subset=["PLAYER_ID", "TEAM_ID", "GAME_DATE"])
    players = players.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort")

    player_timelines: dict[str, tuple[list[np.datetime64], list[str]]] = {}
    for player_id, group in players.groupby("PLAYER_ID", sort=False):
        dates = list(group["GAME_DATE"].to_numpy(dtype="datetime64[ns]"))
        teams = group["TEAM_ID"].astype(str).tolist()
        player_timelines[str(player_id)] = (dates, teams)

    def lookup(player_id: str, date) -> str | None:
        timeline = player_timelines.get(str(player_id))
        if timeline is None:
            return None

        dates, teams = timeline
        date_np = np.datetime64(pd.Timestamp(date).to_datetime64(), "ns")
        idx = bisect_left(dates, date_np) - 1
        if idx < 0:
            return None
        return teams[idx]

    return lookup


def _empty_all_star_voting_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["season_year", "player_id", "team_name", "fan_votes", "score"]
    )


def _normalize_all_star_voting_df(
    all_star_voting_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if all_star_voting_df is None or all_star_voting_df.empty:
        return _empty_all_star_voting_df()

    out = all_star_voting_df.copy()
    out.columns = out.columns.str.lower()
    required_cols = {"season_year", "player_id", "team_name", "fan_votes"}
    missing_cols = sorted(required_cols - set(out.columns))
    if missing_cols:
        raise ValueError(f"all_star_voting_df is missing columns: {missing_cols}")
    if "score" not in out.columns:
        out["score"] = np.nan

    out = out[["season_year", "player_id", "team_name", "fan_votes", "score"]].copy()
    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype(
        "Int64"
    )
    out["player_id"] = out["player_id"].astype(str)
    out["team_name"] = out["team_name"].astype("string")
    out["fan_votes"] = pd.to_numeric(out["fan_votes"], errors="coerce").fillna(0)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["season_year", "player_id"])


def _prepare_players_for_lookup(df_players: pd.DataFrame) -> pd.DataFrame:
    out = df_players.copy()
    for col in ["PLAYER_ID", "TEAM_ID", "GAME_ID", "SEASON_ID"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "SEASON_YEAR" in out.columns:
        out["SEASON_YEAR"] = pd.to_numeric(out["SEASON_YEAR"], errors="coerce")
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    return out


def _precompute_all_star_indexes(all_star_voting_df: pd.DataFrame) -> dict[str, dict]:
    indexes: dict[str, dict] = {}
    for season_year, season_df in all_star_voting_df.groupby("season_year"):
        season_year_int = int(season_year)
        player_votes = (
            season_df.groupby("player_id", sort=False)["fan_votes"].max().to_dict()
        )
        player_scores = season_df.groupby("player_id", sort=False)["score"].min()
        player_scores = player_scores.dropna().to_dict()
        team_players = {
            str(team_name): set(group["player_id"].astype(str))
            for team_name, group in season_df.dropna(subset=["team_name"]).groupby(
                "team_name"
            )
        }
        indexes[season_year_int] = {
            "team_players": team_players,
            "all_player_ids": set(season_df["player_id"].astype(str)),
            "player_votes": {str(k): float(v) for k, v in player_votes.items()},
            "player_scores": {str(k): float(v) for k, v in player_scores.items()},
            "total_fan_votes": float(season_df["fan_votes"].fillna(0).sum()),
        }
    return indexes


def _normalize_injured_dict(
    injured_dict: dict | None,
) -> dict[str, dict[str, set[str]]]:
    """Cast all (game_id, team_id, player_id) keys/values to str exactly once.

    Callers may pass mixed int/str keys depending on the upstream DataFrame dtypes.
    Normalizing here means the row loop can do pure str lookups without per-key fallbacks.
    """
    if not injured_dict:
        return {}

    normalized: dict[str, dict[str, set[str]]] = {}
    for game_id, team_map in injured_dict.items():
        if game_id is None or pd.isna(game_id):
            continue
        normalized_team_map: dict[str, set[str]] = {}
        for team_id, player_ids in (team_map or {}).items():
            if team_id is None or pd.isna(team_id):
                continue
            normalized_team_map[str(team_id)] = {
                str(player_id)
                for player_id in player_ids
                if player_id is not None and not pd.isna(player_id)
            }
        normalized[str(game_id)] = normalized_team_map
    return normalized


def add_all_star_voting_features(
    df_team: pd.DataFrame,
    df_players: pd.DataFrame,
    all_star_voting_df: pd.DataFrame,
    injured_dict: dict,
) -> pd.DataFrame:
    """Adds per-team-game all-star fan-vote share columns.

    Adds columns (all suffixed _BEFORE so select_train_columns keeps them):
        ALL_STAR_FAN_VOTE_SHARE_BEFORE     float
        ALL_STAR_MIN_SCORE_BEFORE          float
        ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE float
        ALL_STAR_MIN_INJURED_SCORE_BEFORE  float
        ALL_STAR_FAN_VOTES_BEFORE          float
        ALL_STAR_CANDIDATE_COUNT_BEFORE    int
        ALL_STAR_SEASON_YEAR_BEFORE        int (audit; ok to keep)
    """
    required_team_cols = {"GAME_ID", "TEAM_ID", "SEASON_ID", "GAME_DATE"}
    missing_team_cols = sorted(required_team_cols - set(df_team.columns))
    if missing_team_cols:
        raise ValueError(f"df_team is missing columns: {missing_team_cols}")

    out = df_team.copy()
    players = _prepare_players_for_lookup(df_players)
    all_star = _normalize_all_star_voting_df(all_star_voting_df)
    all_star_indexes = _precompute_all_star_indexes(all_star)
    injured_dict_normalized = _normalize_injured_dict(injured_dict)

    required_season_years = {
        all_star_season_year_for_game_date(d) for d in out["GAME_DATE"]
    }
    missing_season_years = sorted(
        year
        for year in required_season_years
        if all_star_indexes.get(year) is None
        or all_star_indexes[year]["total_fan_votes"] <= 0
    )
    if missing_season_years:
        raise ValueError(
            "all_star_voting_df is missing usable rows for required season_year(s): "
            f"{missing_season_years}. Fetch/scrape the corresponding All-Star voting "
            "data from Basketball Reference, rebuild the combined all-star voting CSV, "
            "and upload it to Supabase before running this pipeline."
        )

    player_lookup = create_player_lookup(players, injured_dict=injured_dict)
    last_team_lookup = build_last_team_before_date_lookup(players)

    updates = []
    team_rows = out[["GAME_ID", "TEAM_ID", "SEASON_ID", "GAME_DATE"]]
    for game_id, team_id, season_id, game_date in tqdm(
        team_rows.itertuples(index=False, name=None),
        total=len(team_rows),
        desc="Adding all-star voting data",
    ):
        game_date = pd.Timestamp(game_date)
        team_id_str = str(team_id)
        all_star_season_year = all_star_season_year_for_game_date(game_date)
        season_index = all_star_indexes[all_star_season_year]

        row_update = {
            "ALL_STAR_FAN_VOTE_SHARE_BEFORE": np.nan,
            "ALL_STAR_MIN_SCORE_BEFORE": np.nan,
            "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE": np.nan,
            "ALL_STAR_MIN_INJURED_SCORE_BEFORE": np.nan,
            "ALL_STAR_FAN_VOTES_BEFORE": np.nan,
            "ALL_STAR_CANDIDATE_COUNT_BEFORE": 0,
            "ALL_STAR_SEASON_YEAR_BEFORE": all_star_season_year,
        }

        team_name = TEAM_NAME_BY_ID.get(team_id_str)
        candidate_ids: set[str] = set()
        if team_name is not None:
            for player_id in season_index["team_players"].get(team_name, set()):
                last_team_id = last_team_lookup(player_id, game_date)
                if last_team_id is None or str(last_team_id) == team_id_str:
                    candidate_ids.add(str(player_id))

        current_players = player_lookup(
            str(season_id),
            team_id_str,
            game_date,
            game_id=str(game_id),
        )
        if not current_players.empty and "PLAYER_ID" in current_players.columns:
            current_player_ids = set(current_players["PLAYER_ID"].astype(str))
            candidate_ids.update(
                player_id
                for player_id in current_player_ids
                if player_id in season_index["all_player_ids"]
            )

        fan_votes = sum(
            season_index["player_votes"].get(player_id, 0.0)
            for player_id in candidate_ids
        )
        row_update["ALL_STAR_FAN_VOTES_BEFORE"] = fan_votes
        row_update["ALL_STAR_CANDIDATE_COUNT_BEFORE"] = len(candidate_ids)
        row_update["ALL_STAR_FAN_VOTE_SHARE_BEFORE"] = (
            fan_votes / season_index["total_fan_votes"]
        )
        candidate_scores = [
            season_index["player_scores"][player_id]
            for player_id in candidate_ids
            if player_id in season_index["player_scores"]
        ]
        if candidate_scores:
            row_update["ALL_STAR_MIN_SCORE_BEFORE"] = min(candidate_scores)

        injured_ids = injured_dict_normalized.get(str(game_id), {}).get(
            team_id_str, set()
        )
        injured_fan_votes = [
            season_index["player_votes"][player_id]
            for player_id in injured_ids
            if player_id in season_index["player_votes"]
        ]
        row_update["ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE"] = (
            max(injured_fan_votes) / season_index["total_fan_votes"]
            if injured_fan_votes
            else 0.0
        )
        injured_scores = [
            season_index["player_scores"][player_id]
            for player_id in injured_ids
            if player_id in season_index["player_scores"]
        ]
        if injured_scores:
            row_update["ALL_STAR_MIN_INJURED_SCORE_BEFORE"] = min(injured_scores)
        updates.append(row_update)

    updates_df = pd.DataFrame(updates, index=out.index, columns=FEATURE_COLUMNS)
    return pd.concat([out, updates_df], axis=1)
