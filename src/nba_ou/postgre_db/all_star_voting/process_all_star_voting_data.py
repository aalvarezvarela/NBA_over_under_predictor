from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from nba_api.stats.static import players as nba_static_players

from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.players.fetch_players_data_from_db import load_players_from_db

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data/all_star_voting/all_star_voting_combined.csv"
DEFAULT_INJURY_DATA_DIR = PROJECT_ROOT / "data/injury_data"

SOURCE_COLUMNS = [
    "conference",
    "position",
    "season",
    "player_name",
    "fan_votes",
    "fan_rank",
    "player_votes",
    "player_rank",
    "media_votes",
    "media_rank",
    "score",
]
OUTPUT_COLUMNS = [
    "conference",
    "position",
    "season",
    "season_year",
    "player_name",
    "player_id",
    "team_name",
    "fan_votes",
    "fan_votes_pct",
    "fan_rank",
    "player_votes",
    "player_rank",
    "media_votes",
    "media_rank",
    "score",
]
NUMERIC_COLUMNS = [
    "fan_votes",
    "fan_rank",
    "player_votes",
    "player_rank",
    "media_votes",
    "media_rank",
    "score",
]
SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
SPECIAL_NAME_TRANSLITERATION = str.maketrans({"ё": "e", "Ё": "E", "đ": "dj", "Đ": "Dj"})

# Kyle Mangas is present in the local NBA injury feed as player_id 1630667, but
# is absent from the bundled nba_api static player list and PlayerIndex response.
KNOWN_PLAYER_ID_OVERRIDES = {"kyle mangas": "1630667"}
KNOWN_TEAM_OVERRIDES = {
    # RJ Nembhard Jr. appears in the 2022-23 voting file, but the local
    # Supabase game logs have no non-preseason Cavaliers row for him before
    # the February 15 cutoff. Added 2026-06-08 to avoid using preseason as a
    # generic fallback source.
    (2022, "1630612"): "Cleveland Cavaliers",
    # Kyle Mangas is present in the 2025-26 voting file before he appears in
    # local non-preseason player logs. Added 2026-06-08 from local injury data.
    (2025, "1630667"): "San Antonio Spurs",
}
TEAM_NAME_BY_ID = {
    str(team_id): team_name for team_name, team_id in TEAM_ID_MAP.items()
}


@dataclass(frozen=True)
class PlayerMatchErrorDetail:
    season: str
    player_name: str
    reason: str
    candidate_player_ids: tuple[str, ...] = ()


class PlayerMatchError(ValueError):
    def __init__(self, details: list[PlayerMatchErrorDetail]) -> None:
        self.details = details
        preview = "\n".join(_format_error_detail(detail) for detail in details[:25])
        extra = ""
        if len(details) > 25:
            extra = f"\n... and {len(details) - 25} more"
        super().__init__(
            f"Could not resolve all all-star voting player IDs:\n{preview}{extra}"
        )


def _format_error_detail(detail: PlayerMatchErrorDetail) -> str:
    if detail.candidate_player_ids:
        ids = ", ".join(detail.candidate_player_ids)
        return (
            f"- {detail.season}: {detail.player_name} "
            f"({detail.reason}; candidates: {ids})"
        )
    return f"- {detail.season}: {detail.player_name} ({detail.reason})"


def _name_tokens(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    normalized = str(value).translate(SPECIAL_NAME_TRANSLITERATION)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower().replace(".", "").replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return [token for token in normalized.split() if token]


def _strip_suffix_tokens(tokens: list[str]) -> list[str]:
    stripped = list(tokens)
    while stripped and stripped[-1] in SUFFIX_TOKENS:
        stripped.pop()
    return stripped


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def _exact_name_keys(name: object) -> list[str]:
    tokens = _name_tokens(name)
    keys = []
    for variant in (tokens, _strip_suffix_tokens(tokens)):
        if variant:
            keys.append(_join_tokens(variant))
    return list(dict.fromkeys(keys))


def _display_name_keys(name: object) -> list[str]:
    tokens = _name_tokens(name)
    keys = _exact_name_keys(name)
    stripped = _strip_suffix_tokens(tokens)
    if len(stripped) >= 2:
        keys.append(_join_tokens([stripped[0][0], *stripped[1:]]))
    return list(dict.fromkeys(key for key in keys if key))


def previous_season(season: str) -> str:
    start_year = int(season[:4])
    return f"{start_year - 1}-{str(start_year)[-2:]}"


def season_start_year(season: str) -> int:
    return int(season[:4])


def all_star_team_cutoff_date(season_year: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(season_year) + 1, month=2, day=15)


def required_player_seasons(voting_df: pd.DataFrame) -> list[str]:
    seasons = set(voting_df["season"].dropna().astype(str))
    seasons.update(previous_season(season) for season in list(seasons))
    return sorted(seasons)


def season_label_from_year(season_year: int) -> str:
    return f"{season_year}_{str(season_year + 1)[-2:]}"


def load_injury_reports_for_season_years(
    season_years: list[int],
    injury_data_dir: Path = DEFAULT_INJURY_DATA_DIR,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for season_year in sorted(set(int(year) for year in season_years)):
        path = (
            injury_data_dir / f"nba_injuries_{season_label_from_year(season_year)}.csv"
        )
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        df["season_year"] = season_year
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(
            columns=["season_year", "player_id", "team_id", "game_date"]
        )
    return pd.concat(dfs, ignore_index=True, sort=False)


def read_voting_csv(input_csv: Path = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{input_csv} is missing required columns: {missing_columns}")

    out = df[SOURCE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["season"].isna().any():
        raise ValueError("All rows must have a season value.")
    if out["player_name"].isna().any():
        raise ValueError("All rows must have a player_name value.")

    out["season"] = out["season"].astype(str)
    out["season_year"] = out["season"].map(season_start_year)
    return out


def add_fan_votes_pct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season_year" not in out.columns:
        out["season_year"] = out["season"].astype(str).map(season_start_year)
    season_totals = out.groupby("season_year")["fan_votes"].transform("sum")
    zero_total_seasons = sorted(out.loc[season_totals.eq(0), "season"].unique())
    if zero_total_seasons:
        raise ValueError(
            "Cannot calculate fan vote percentages for seasons with zero total "
            f"fan votes: {zero_total_seasons}"
        )
    out["fan_votes_pct"] = out["fan_votes"] / season_totals
    return out


def _standardize_team_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    candidate = str(value).strip()
    if not candidate:
        return None

    mapped = TEAM_NAME_STANDARDIZATION.get(candidate, candidate)
    if mapped is None:
        return None
    return str(mapped)


def _team_name_from_log_row(row: pd.Series) -> str | None:
    team_id = str(row.get("team_id", "")).strip()
    if team_id in TEAM_NAME_BY_ID:
        return _standardize_team_name(TEAM_NAME_BY_ID[team_id])

    team_city = row.get("team_city")
    team_name = row.get("team_name")
    if pd.notna(team_city) and pd.notna(team_name):
        standardized = _standardize_team_name(f"{team_city} {team_name}")
        if standardized is not None:
            return standardized

    return _standardize_team_name(team_city)


def add_team_names_at_cutoff(
    voting_df: pd.DataFrame,
    players_df: pd.DataFrame,
    games_df: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    *,
    skip_unresolved: bool = False,
) -> pd.DataFrame:
    out = voting_df.copy()
    required_player_cols = {
        "season_year",
        "player_id",
        "game_id",
        "team_id",
        "team_city",
        "team_name",
    }
    missing_player_cols = sorted(required_player_cols - set(players_df.columns))
    if missing_player_cols:
        raise ValueError(f"players_df is missing columns: {missing_player_cols}")

    required_game_cols = {"game_id", "team_id", "game_date", "season_type"}
    missing_game_cols = sorted(required_game_cols - set(games_df.columns))
    if missing_game_cols:
        raise ValueError(f"games_df is missing columns: {missing_game_cols}")

    player_cols = [
        "season_year",
        "player_id",
        "game_id",
        "team_id",
        "team_city",
        "team_name",
    ]
    player_logs = players_df[player_cols].copy()
    player_logs["player_id"] = player_logs["player_id"].astype(str)
    player_logs["team_id"] = player_logs["team_id"].astype(str)
    player_logs["game_id"] = player_logs["game_id"].astype(str)
    player_logs["season_year"] = pd.to_numeric(
        player_logs["season_year"], errors="coerce"
    ).astype("Int64")

    game_cols = ["game_id", "team_id", "game_date", "season_type"]
    game_dates = games_df[game_cols].drop_duplicates().copy()
    game_dates["game_id"] = game_dates["game_id"].astype(str)
    game_dates["team_id"] = game_dates["team_id"].astype(str)
    game_dates["game_date"] = pd.to_datetime(
        game_dates["game_date"], errors="coerce", format="mixed"
    )
    all_game_dates = game_dates.copy()
    season_type = game_dates["season_type"].astype("string").str.lower()
    game_dates = game_dates[
        ~season_type.str.contains("pre", na=False)
        & ~season_type.str.contains("all", na=False)
    ]

    logs = player_logs.merge(
        game_dates[["game_id", "team_id", "game_date"]],
        on=["game_id", "team_id"],
        how="left",
    ).dropna(subset=["season_year", "game_date"])
    team_lookup: dict[tuple[int, str], str] = {}
    if not logs.empty:
        logs["cutoff_date"] = logs["season_year"].map(all_star_team_cutoff_date)
        logs = logs[logs["game_date"] <= logs["cutoff_date"]].copy()

        logs["team_name_standardized"] = logs.apply(_team_name_from_log_row, axis=1)
        logs = logs.dropna(subset=["team_name_standardized"])
        logs = logs.sort_values(
            ["season_year", "player_id", "game_date", "game_id"],
            kind="mergesort",
        )
        latest_logs = logs.groupby(["season_year", "player_id"], as_index=False).tail(1)
        team_lookup = {
            (int(row.season_year), str(row.player_id)): row.team_name_standardized
            for row in latest_logs.itertuples(index=False)
        }

    fallback_player_lookup = _build_player_team_lookup_before_cutoff(
        player_logs, all_game_dates
    )
    injury_lookup = _build_injury_team_lookup(injuries_df)

    def resolve_team(row: pd.Series) -> str | None:
        key = (int(row["season_year"]), str(row["player_id"]))
        team = (
            team_lookup.get(key)
            or injury_lookup.get(key)
            or fallback_player_lookup.get(key)
        )
        if team is not None:
            return team
        override = KNOWN_TEAM_OVERRIDES.get(key)
        if override is not None:
            logger.warning(
                "Using hardcoded team override for season_year=%s player_id=%s -> %s",
                key[0],
                key[1],
                override,
            )
        return override

    out["team_name"] = out.apply(resolve_team, axis=1)
    missing = out[out["team_name"].isna()][
        ["season", "season_year", "player_name", "player_id"]
    ].drop_duplicates()
    if not missing.empty:
        if skip_unresolved:
            for row in missing.itertuples(index=False):
                logger.warning(
                    "Skipping all-star voting row with unresolved team: "
                    "season=%s player_name=%s player_id=%s",
                    row.season,
                    row.player_name,
                    row.player_id,
                )
            return out[out["team_name"].notna()].copy()

        details = "\n".join(
            f"- {row.season}: {row.player_name} ({row.player_id})"
            for row in missing.itertuples(index=False)
        )
        raise ValueError(
            "Could not resolve all-star voting team names as of February 15:\n"
            f"{details}"
        )

    return out


def _build_player_team_lookup_before_cutoff(
    player_logs: pd.DataFrame,
    game_dates: pd.DataFrame,
) -> dict[tuple[int, str], str]:
    logs = player_logs.merge(
        game_dates[["game_id", "team_id", "game_date", "season_type"]],
        on=["game_id", "team_id"],
        how="left",
    ).dropna(subset=["season_year", "game_date"])
    if logs.empty:
        return {}

    season_type = logs["season_type"].astype("string").str.lower()
    logs = logs[
        ~season_type.str.contains("pre", na=False)
        & ~season_type.str.contains("all", na=False)
    ].copy()
    logs["cutoff_date"] = logs["season_year"].map(all_star_team_cutoff_date)
    logs = logs[logs["game_date"] <= logs["cutoff_date"]].copy()
    if logs.empty:
        return {}

    logs["team_name_standardized"] = logs.apply(_team_name_from_log_row, axis=1)
    logs = logs.dropna(subset=["team_name_standardized"])
    logs = logs.sort_values(
        ["season_year", "player_id", "game_date", "game_id"],
        kind="mergesort",
    )
    latest_logs = logs.groupby(["season_year", "player_id"], as_index=False).tail(1)
    return {
        (int(row.season_year), str(row.player_id)): row.team_name_standardized
        for row in latest_logs.itertuples(index=False)
    }


def _build_injury_team_lookup(
    injuries_df: pd.DataFrame | None,
) -> dict[tuple[int, str], str]:
    if injuries_df is None or injuries_df.empty:
        return {}

    injuries = injuries_df.copy()
    injuries.columns = injuries.columns.str.lower()
    required_cols = {"season_year", "player_id", "team_id", "game_date"}
    missing_cols = sorted(required_cols - set(injuries.columns))
    if missing_cols:
        raise ValueError(f"injuries_df is missing columns: {missing_cols}")

    injuries["season_year"] = pd.to_numeric(
        injuries["season_year"], errors="coerce"
    ).astype("Int64")
    injuries["player_id"] = injuries["player_id"].astype(str)
    injuries["team_id"] = injuries["team_id"].astype(str)
    injuries["game_date"] = pd.to_datetime(
        injuries["game_date"], errors="coerce", format="mixed"
    )
    injuries = injuries.dropna(subset=["season_year", "player_id", "game_date"])
    if injuries.empty:
        return {}

    injuries["cutoff_date"] = injuries["season_year"].map(all_star_team_cutoff_date)
    injuries = injuries[injuries["game_date"] <= injuries["cutoff_date"]].copy()
    if injuries.empty:
        return {}

    injuries["team_name_standardized"] = injuries.apply(_team_name_from_log_row, axis=1)
    injuries = injuries.dropna(subset=["team_name_standardized"])
    injuries = injuries.sort_values(
        ["season_year", "player_id", "game_date", "team_id"],
        kind="mergesort",
    )
    latest_injuries = injuries.groupby(
        ["season_year", "player_id"], as_index=False
    ).tail(1)
    return {
        (int(row.season_year), str(row.player_id)): row.team_name_standardized
        for row in latest_injuries.itertuples(index=False)
    }


def _full_name_series(players_df: pd.DataFrame) -> pd.Series:
    first = players_df.get("firstname", pd.Series("", index=players_df.index))
    family = players_df.get("familyname", pd.Series("", index=players_df.index))
    return (
        first.fillna("").astype(str).str.strip()
        + " "
        + family.fillna("").astype(str).str.strip()
    ).str.strip()


class AllStarVotingPlayerMatcher:
    def __init__(self, players_df: pd.DataFrame) -> None:
        required_cols = {"season_year", "player_id", "player_name"}
        missing_cols = sorted(required_cols - set(players_df.columns))
        if missing_cols:
            raise ValueError(f"players_df is missing required columns: {missing_cols}")

        self.players_df = players_df.copy()
        self.players_df["player_id"] = self.players_df["player_id"].astype(str)
        self.players_df["full_name"] = _full_name_series(self.players_df)
        self.global_full_keys_by_id = self._build_global_full_keys_by_id()
        (
            self.nba_static_full_index,
            self.nba_static_display_index,
            self.nba_static_tokens_by_id,
        ) = self._build_nba_static_indexes()
        self._add_nba_static_full_name_keys()

    def _build_global_full_keys_by_id(self) -> dict[str, set[str]]:
        keys_by_id: dict[str, set[str]] = {}
        rows = self.players_df[["player_id", "full_name"]].drop_duplicates()
        for row in rows.itertuples(index=False):
            if not row.full_name:
                continue
            keys = _exact_name_keys(row.full_name)
            if keys:
                keys_by_id.setdefault(str(row.player_id), set()).update(keys)
        return keys_by_id

    def _add_nba_static_full_name_keys(self) -> None:
        supabase_player_ids = set(self.players_df["player_id"].dropna().astype(str))
        for player in nba_static_players.get_players():
            player_id = str(player.get("id", ""))
            if player_id not in supabase_player_ids:
                continue
            for key in _exact_name_keys(player.get("full_name")):
                self.global_full_keys_by_id.setdefault(player_id, set()).add(key)

    def _build_nba_static_indexes(
        self,
    ) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, list[str]]]:
        full_index: dict[str, set[str]] = {}
        display_index: dict[str, set[str]] = {}
        tokens_by_id: dict[str, list[str]] = {}

        for player in nba_static_players.get_players():
            player_id = str(player.get("id", ""))
            full_name = player.get("full_name")
            tokens_by_id[player_id] = _name_tokens(full_name)
            for key in _exact_name_keys(full_name):
                full_index.setdefault(key, set()).add(player_id)
            for key in _display_name_keys(full_name):
                display_index.setdefault(key, set()).add(player_id)

        return full_index, display_index, tokens_by_id

    def match_voting_players(self, voting_df: pd.DataFrame) -> pd.Series:
        player_ids = pd.Series(index=voting_df.index, dtype="object")
        details: list[PlayerMatchErrorDetail] = []

        for season, season_df in voting_df.groupby("season", sort=True):
            start_year = season_start_year(str(season))
            previous_year = season_start_year(previous_season(str(season)))
            season_players = self.players_df[
                self.players_df["season_year"].isin([start_year, previous_year])
            ]
            full_index, display_index = self._build_season_indexes(season_players)

            season_name_to_id: dict[str, str] = {}
            for player_name in sorted(season_df["player_name"].unique()):
                player_id, error = self._resolve_name(
                    str(season), str(player_name), full_index, display_index
                )
                if error is not None:
                    details.append(error)
                    continue
                season_name_to_id[str(player_name)] = player_id

            player_ids.loc[season_df.index] = season_df["player_name"].map(
                season_name_to_id
            )

        if details:
            raise PlayerMatchError(details)

        if player_ids.isna().any():
            unresolved = voting_df.loc[player_ids.isna(), ["season", "player_name"]]
            details = [
                PlayerMatchErrorDetail(
                    season=str(row.season),
                    player_name=str(row.player_name),
                    reason="unresolved after matching",
                )
                for row in unresolved.drop_duplicates().itertuples(index=False)
            ]
            raise PlayerMatchError(details)

        return player_ids

    def _build_season_indexes(
        self, season_players: pd.DataFrame
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        full_index: dict[str, set[str]] = {}
        display_index: dict[str, set[str]] = {}
        rows = season_players[
            ["player_id", "player_name", "full_name"]
        ].drop_duplicates()

        for row in rows.itertuples(index=False):
            player_id = str(row.player_id)
            for key in _exact_name_keys(row.full_name):
                full_index.setdefault(key, set()).add(player_id)
            for key in _exact_name_keys(row.player_name):
                display_index.setdefault(key, set()).add(player_id)

        return full_index, display_index

    def _resolve_name(
        self,
        season: str,
        player_name: str,
        full_index: dict[str, set[str]],
        display_index: dict[str, set[str]],
    ) -> tuple[str | None, PlayerMatchErrorDetail | None]:
        full_keys = _exact_name_keys(player_name)
        full_ids = _lookup_ids(full_index, full_keys)
        if len(full_ids) == 1:
            return next(iter(full_ids)), None
        if len(full_ids) > 1:
            filtered_ids = self._filter_ids_by_global_full_name(full_ids, full_keys)
            if len(filtered_ids) == 1:
                return next(iter(filtered_ids)), None
            return None, PlayerMatchErrorDetail(
                season=season,
                player_name=player_name,
                reason="ambiguous full-name match",
                candidate_player_ids=tuple(sorted(full_ids)),
            )

        display_ids = _lookup_ids(display_index, _display_name_keys(player_name))
        if len(display_ids) == 1:
            return next(iter(display_ids)), None
        if len(display_ids) > 1:
            filtered_ids = self._filter_ids_by_global_full_name(display_ids, full_keys)
            if len(filtered_ids) == 1:
                return next(iter(filtered_ids)), None
            return None, PlayerMatchErrorDetail(
                season=season,
                player_name=player_name,
                reason="ambiguous display-name match",
                candidate_player_ids=tuple(sorted(display_ids)),
            )

        reversed_ids = self._lookup_ids_by_reversed_full_name(
            display_index, player_name
        )
        if len(reversed_ids) == 1:
            return next(iter(reversed_ids)), None
        if len(reversed_ids) > 1:
            return None, PlayerMatchErrorDetail(
                season=season,
                player_name=player_name,
                reason="ambiguous reversed full-name match",
                candidate_player_ids=tuple(sorted(reversed_ids)),
            )

        nba_static_ids = self._lookup_nba_static_ids(player_name)
        if len(nba_static_ids) == 1:
            return next(iter(nba_static_ids)), None
        if len(nba_static_ids) > 1:
            return None, PlayerMatchErrorDetail(
                season=season,
                player_name=player_name,
                reason="ambiguous NBA API fallback match",
                candidate_player_ids=tuple(sorted(nba_static_ids)),
            )

        override_id = self._lookup_known_override(player_name)
        if override_id is not None:
            return override_id, None

        return None, PlayerMatchErrorDetail(
            season=season,
            player_name=player_name,
            reason=(
                "not found in Supabase season-pair data, NBA API static data, "
                "or known overrides"
            ),
        )

    def _filter_ids_by_global_full_name(
        self, player_ids: set[str], full_keys: list[str]
    ) -> set[str]:
        query_keys = set(full_keys)
        return {
            player_id
            for player_id in player_ids
            if self.global_full_keys_by_id.get(player_id, set()) & query_keys
        }

    def _lookup_ids_by_reversed_full_name(
        self, display_index: dict[str, set[str]], player_name: str
    ) -> set[str]:
        tokens = _strip_suffix_tokens(_name_tokens(player_name))
        if len(tokens) != 2:
            return set()

        reversed_key = _join_tokens([tokens[1], tokens[0]])
        display_key = _join_tokens([tokens[1][0], tokens[0]])
        candidate_ids = display_index.get(display_key, set())
        return {
            player_id
            for player_id in candidate_ids
            if reversed_key in self.global_full_keys_by_id.get(player_id, set())
        }

    def _lookup_nba_static_ids(self, player_name: str) -> set[str]:
        full_ids = _lookup_ids(
            self.nba_static_full_index, _exact_name_keys(player_name)
        )
        if full_ids:
            return full_ids

        display_ids = _lookup_ids(
            self.nba_static_display_index, _display_name_keys(player_name)
        )
        if len(display_ids) <= 1:
            return display_ids

        suffix_filtered_ids = self._filter_static_ids_by_query_suffix(
            display_ids, player_name
        )
        return suffix_filtered_ids or display_ids

    def _filter_static_ids_by_query_suffix(
        self, player_ids: set[str], player_name: str
    ) -> set[str]:
        query_tokens = _name_tokens(player_name)
        query_suffixes = [token for token in query_tokens if token in SUFFIX_TOKENS]
        if not query_suffixes:
            return set()

        query_suffix = query_suffixes[-1]
        return {
            player_id
            for player_id in player_ids
            if self.nba_static_tokens_by_id.get(player_id, [])
            and self.nba_static_tokens_by_id[player_id][-1] == query_suffix
        }

    def _lookup_known_override(self, player_name: str) -> str | None:
        keys = _exact_name_keys(player_name)
        for key in keys:
            if key in KNOWN_PLAYER_ID_OVERRIDES:
                override_id = KNOWN_PLAYER_ID_OVERRIDES[key]
                logger.warning(
                    "Using hardcoded player_id override for %r -> %s",
                    player_name,
                    override_id,
                )
                return override_id
        return None


def _lookup_ids(index: dict[str, set[str]], keys: list[str]) -> set[str]:
    ids: set[str] = set()
    for key in keys:
        ids.update(index.get(key, set()))
    return ids


def prepare_all_star_voting_dataset(
    input_csv: Path = DEFAULT_INPUT_CSV,
    players_df: pd.DataFrame | None = None,
    games_df: pd.DataFrame | None = None,
    injuries_df: pd.DataFrame | None = None,
    *,
    skip_unresolved: bool = False,
) -> pd.DataFrame:
    voting_df = read_voting_csv(input_csv)
    voting_df = add_fan_votes_pct(voting_df)

    if players_df is None:
        seasons = required_player_seasons(voting_df)
        print(f"Loading Supabase player data for seasons: {', '.join(seasons)}")
        players_df = load_players_from_db(seasons=seasons)
        if players_df is None:
            raise RuntimeError("Failed to load players data from Supabase.")

    matcher = AllStarVotingPlayerMatcher(players_df)
    try:
        voting_df["player_id"] = matcher.match_voting_players(voting_df)
    except PlayerMatchError as exc:
        if not skip_unresolved:
            raise
        unresolved_pairs = {
            (str(detail.season), str(detail.player_name)) for detail in exc.details
        }
        logger.warning(
            "Skipping %d unresolved all-star voting player(s): %s",
            len(unresolved_pairs),
            ", ".join(f"{season}: {name}" for season, name in sorted(unresolved_pairs)),
        )
        unresolved_mask = voting_df.apply(
            lambda row: (
                (
                    str(row["season"]),
                    str(row["player_name"]),
                )
                in unresolved_pairs
            ),
            axis=1,
        )
        voting_df = voting_df[~unresolved_mask].copy()
        if voting_df.empty:
            return voting_df.reindex(columns=OUTPUT_COLUMNS)
        voting_df["player_id"] = matcher.match_voting_players(voting_df)

    if games_df is None:
        seasons = sorted(voting_df["season"].dropna().astype(str).unique())
        print(f"Loading Supabase games data for seasons: {', '.join(seasons)}")
        games_df = load_games_from_db(seasons=seasons)
        if games_df is None:
            raise RuntimeError("Failed to load games data from Supabase.")

    if injuries_df is None:
        injuries_df = load_injury_reports_for_season_years(
            sorted(voting_df["season_year"].dropna().astype(int).unique().tolist())
        )

    voting_df = add_team_names_at_cutoff(
        voting_df,
        players_df,
        games_df,
        injuries_df=injuries_df,
        skip_unresolved=skip_unresolved,
    )
    return voting_df[OUTPUT_COLUMNS]
