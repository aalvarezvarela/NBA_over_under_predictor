from pathlib import Path

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_cleaned_games_for_odds,
)
from tqdm import tqdm

MATCHUP_HISTORY_BASE_COLUMNS = [
    "game_date",
    "season_year",
    "event_id",
    "start_time",
    "matchup_url",
    "team_away",
    "team_home",
]

MATCHUP_HISTORY_NUMERIC_COLUMNS = [
    "matchup_offensive_team_records_points_for_per_game_away",
    "matchup_offensive_team_records_points_for_per_game_home",
    "matchup_offensive_team_records_points_against_per_game_away",
    "matchup_offensive_team_records_points_against_per_game_home",
    "matchup_offensive_team_records_avg_1st_half_points_away",
    "matchup_offensive_team_records_avg_1st_half_points_home",
    "matchup_offensive_team_records_offensive_fg_pct_away",
    "matchup_offensive_team_records_offensive_fg_pct_home",
    "matchup_offensive_team_records_offensive_ft_pct_away",
    "matchup_offensive_team_records_offensive_ft_pct_home",
    "matchup_offensive_team_records_offensive_3pt_pct_away",
    "matchup_offensive_team_records_offensive_3pt_pct_home",
    "matchup_offensive_team_records_offensive_reb_away",
    "matchup_offensive_team_records_offensive_reb_home",
    "matchup_offensive_team_records_avg_fg_made_away",
    "matchup_offensive_team_records_avg_fg_made_home",
    "matchup_offensive_team_records_avg_ft_made_away",
    "matchup_offensive_team_records_avg_ft_made_home",
    "matchup_offensive_team_records_avg_3pt_made_away",
    "matchup_offensive_team_records_avg_3pt_made_home",
    "matchup_defensive_team_records_opponent_points_away",
    "matchup_defensive_team_records_opponent_points_home",
    "matchup_defensive_team_records_opponent_reb_away",
    "matchup_defensive_team_records_opponent_reb_home",
    "matchup_defensive_team_records_opponent_fg_pct_away",
    "matchup_defensive_team_records_opponent_fg_pct_home",
    "matchup_defensive_team_records_opponent_ft_pct_away",
    "matchup_defensive_team_records_opponent_ft_pct_home",
    "matchup_defensive_team_records_opponent_3pt_pct_away",
    "matchup_defensive_team_records_opponent_3pt_pct_home",
]

MATCHUP_HISTORY_TEXT_COLUMNS = [
    "matchup_offensive_team_records_over_under_push_away",
    "matchup_offensive_team_records_over_under_push_home",
]

MATCHUP_HISTORY_COLUMNS = (
    MATCHUP_HISTORY_BASE_COLUMNS
    + MATCHUP_HISTORY_NUMERIC_COLUMNS
    + MATCHUP_HISTORY_TEXT_COLUMNS
)
MATCHUP_HISTORY_OPTIONAL_ENRICHMENT_COLUMNS = [
    "game_id",
    "game_time_utc",
    "game_start_timestamp_utc",
    "game_start_timestamp",
]


def discover_matchup_history_csvs(
    matchup_history_root_dir: str | Path,
    *,
    season_dir_glob: str = "*",
) -> list[Path]:
    root = Path(matchup_history_root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Matchup-history root dir not found: {root}")

    direct_files = list(root.glob("*_matchup_records.csv"))
    nested_files = list(
        root.glob(f"{season_dir_glob}/matchup_records/*_matchup_records.csv")
    )
    files = sorted(set(direct_files + nested_files))
    if not files:
        raise FileNotFoundError(f"No *_matchup_records.csv files found under {root}")
    return files


def _read_matchup_history_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=MATCHUP_HISTORY_COLUMNS)

    missing_base = [
        col for col in MATCHUP_HISTORY_BASE_COLUMNS if col not in df.columns
    ]
    if missing_base:
        raise ValueError(f"Missing required columns in {path}: {missing_base}")

    out = df.copy()
    for col in MATCHUP_HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    allowed_columns = (
        MATCHUP_HISTORY_COLUMNS + MATCHUP_HISTORY_OPTIONAL_ENRICHMENT_COLUMNS
    )
    unknown_columns = [col for col in out.columns if col not in allowed_columns]
    if unknown_columns:
        raise ValueError(f"Unexpected columns in {path}: {unknown_columns}")

    enrichment_cols = [
        col for col in MATCHUP_HISTORY_OPTIONAL_ENRICHMENT_COLUMNS if col in out.columns
    ]
    return out[MATCHUP_HISTORY_COLUMNS + enrichment_cols]


def _normalize_sbr_team_name(value: object) -> object:
    if pd.isna(value):
        return value

    name = " ".join(str(value).strip().split())
    candidates = [
        name,
        name.title(),
        name.upper(),
        name.replace("LA ", "L.A. ", 1),
        name.title().replace("La ", "L.A. ", 1),
    ]

    for candidate in candidates:
        mapped = TEAM_NAME_STANDARDIZATION.get(candidate)
        if mapped is None and candidate in TEAM_NAME_STANDARDIZATION:
            raise RuntimeError(f"Team name maps to None: {candidate}")
        if mapped:
            return mapped

    raise RuntimeError(f"Unrecognized SBR team name: {name}")


def normalize_matchup_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype(
        "Int64"
    )
    out["event_id"] = out["event_id"].astype(str)
    out["game_start_timestamp_utc"] = _resolve_game_start_timestamp_utc(out)

    for col in MATCHUP_HISTORY_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in MATCHUP_HISTORY_TEXT_COLUMNS:
        out[col] = out[col].replace("-", pd.NA)

    out["team_away"] = out["team_away"].map(_normalize_sbr_team_name)
    out["team_home"] = out["team_home"].map(_normalize_sbr_team_name)

    return out


def _resolve_game_start_timestamp_utc(df: pd.DataFrame) -> pd.Series:
    resolved = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in ["game_time_utc", "game_start_timestamp_utc", "game_start_timestamp"]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        resolved = resolved.combine_first(parsed)

    fallback = pd.to_datetime(
        _build_game_start_timestamp(df), errors="coerce", utc=True
    )
    return resolved.combine_first(fallback)


def _build_game_start_timestamp(df: pd.DataFrame) -> pd.Series:
    start_time = (
        df["start_time"]
        .astype("string")
        .str.replace(r"\s*\(?ET\)?\s*$", "", regex=True)
        .str.strip()
    )
    start_time = start_time.mask(start_time.isin(["", "-", "nan", "NaN"]))
    combined = (df["game_date"].astype(str) + " " + start_time).where(
        start_time.notna()
    )
    return pd.to_datetime(combined, errors="coerce")


def build_matchup_history_df_from_csvs(
    matchup_history_root_dir: str | Path,
    *,
    season_dir_glob: str = "*",
) -> pd.DataFrame:
    csv_paths = discover_matchup_history_csvs(
        matchup_history_root_dir,
        season_dir_glob=season_dir_glob,
    )

    frames: list[pd.DataFrame] = []
    for csv_path in tqdm(csv_paths, desc="Reading matchup-history CSVs", unit="file"):
        day_df = _read_matchup_history_csv(csv_path)
        if not day_df.empty:
            frames.append(day_df)

    if not frames:
        return pd.DataFrame(columns=MATCHUP_HISTORY_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = normalize_matchup_history_df(out)
    out = out.drop_duplicates().reset_index(drop=True)
    print(
        f"Built matchup-history dataframe: {out.shape[0]} rows x {out.shape[1]} columns"
    )
    return out


def build_games_home_away_for_matchup_history(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "game_season_year",
                "team_home",
                "team_away",
            ]
        )

    df = games_df.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["team_name"] = (
        df["team_name"].map(TEAM_NAME_STANDARDIZATION).fillna(df["team_name"])
    )

    is_home = df["home"].astype("boolean")
    home = df[is_home.fillna(False)].rename(
        columns={"team_name": "team_home", "season_year": "game_season_year"}
    )
    away = df[(~is_home).fillna(False)].rename(columns={"team_name": "team_away"})

    home_cols = ["game_id", "game_date", "game_season_year", "team_home"]
    away_cols = ["game_id", "game_date", "team_away"]
    games_ha = home[home_cols].merge(away[away_cols], on=["game_id", "game_date"])
    return games_ha.drop_duplicates(
        subset=["game_date", "team_home", "team_away"],
        keep="first",
    )


def merge_matchup_history_with_games(
    matchup_history_df: pd.DataFrame,
    games_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if matchup_history_df.empty:
        return matchup_history_df.copy()

    out = matchup_history_df.copy()
    out = out.drop(columns=["game_id"], errors="ignore")

    if games_df is None or games_df.empty:
        print("No games data provided; returning matchup-history data without game_id.")
        return out

    games_ha = build_games_home_away_for_matchup_history(games_df)
    if games_ha.empty:
        print("No home/away games mapping found; returning data without game_id.")
        return out

    out = out.merge(
        games_ha,
        on=["game_date", "team_home", "team_away"],
        how="left",
    )
    if "game_season_year" in out.columns:
        out["season_year"] = out["season_year"].fillna(out["game_season_year"])
        out = out.drop(columns=["game_season_year"])
    return out


def load_games_for_matchup_history_creation() -> pd.DataFrame:
    games_df = load_cleaned_games_for_odds(season_year=None)
    if games_df is None:
        return pd.DataFrame()
    return games_df
