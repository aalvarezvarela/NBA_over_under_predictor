from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


def _safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().empty:
        return np.nan
    return float(x.mean())


def _ensure_datetime(df: pd.DataFrame, col: str = "GAME_DATE") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _build_injured_games_index(
    injury_dict: dict[str, dict[str, list[Any]]],
) -> dict[int, dict[int, set]]:
    """
    injury_dict: {GAME_ID: {TEAM_ID: [player_ids...]}}
    returns: {TEAM_ID: {PLAYER_ID: set([GAME_ID, ...])}}
    """
    idx: dict[int, dict[int, set]] = {}
    for game_id_str, team_map in injury_dict.items():
        try:
            game_id = int(game_id_str)
        except Exception:
            # If your GAME_IDs are not int-like, you can keep them as strings.
            # But ensure df GAME_ID matching uses same type.
            continue

        for team_id_str, players in team_map.items():
            try:
                team_id = int(team_id_str)
            except Exception:
                continue

            bucket = idx.setdefault(team_id, {})
            for pid in players:
                if pd.isna(pid) or pid in (0, "0", ""):
                    continue
                try:
                    player_id = int(pid)
                except Exception:
                    continue
                bucket.setdefault(player_id, set()).add(game_id)
    return idx


def _infer_last_two_season_years_for_row(season_year: int) -> tuple[int, int]:
    return (season_year - 1, season_year)


def _get_recent_history_df(
    df_hist: pd.DataFrame,
    *,
    team_id: int,
    season_years: tuple[int, int],
    before_date: pd.Timestamp,
) -> pd.DataFrame:
    y1, y2 = season_years
    mask = (
        (df_hist["TEAM_ID"].astype("int64") == int(team_id))
        & (df_hist["SEASON_YEAR"].astype("int64").isin([y1, y2]))
        & (df_hist["GAME_DATE"] < before_date)
    )
    return df_hist.loc[mask]


def _compute_player_presence_effect(
    df_team_hist: pd.DataFrame,
    injured_games_for_player: set,
) -> tuple[float, float]:
    """
    Returns:
      (mean(TOTAL_POINTS|present) - mean(TOTAL_POINTS|injured),
       mean(DIFF_FROM_LINE|present) - mean(DIFF_FROM_LINE|injured))
    If no injured games found, return (0,0).
    """
    if df_team_hist.empty or not injured_games_for_player:
        return 0.0, 0.0

    game_ids = pd.to_numeric(df_team_hist["GAME_ID"], errors="coerce").astype("Int64")
    df_team_hist = df_team_hist.assign(_GAME_ID_INT=game_ids)

    inj_mask = df_team_hist["_GAME_ID_INT"].isin(list(injured_games_for_player))
    df_inj = df_team_hist.loc[inj_mask]
    if df_inj.empty:
        return 0.0, 0.0

    df_present = df_team_hist.loc[~inj_mask]
    if df_present.empty:
        return 0.0, 0.0

    tot_present = _safe_mean(df_present["TOTAL_POINTS"])
    tot_inj = _safe_mean(df_inj["TOTAL_POINTS"])
    dfl_present = _safe_mean(df_present["DIFF_FROM_LINE"])
    dfl_inj = _safe_mean(df_inj["DIFF_FROM_LINE"])

    if (
        np.isnan(tot_present)
        or np.isnan(tot_inj)
        or np.isnan(dfl_present)
        or np.isnan(dfl_inj)
    ):
        return 0.0, 0.0

    return float(tot_present - tot_inj), float(dfl_present - dfl_inj)


def add_top3_absence_effect_features_for_columns(
    df_games: pd.DataFrame,
    injured_dict: dict[str, dict[str, list[Any]]],
    *,
    home_team_id_col: str = "TEAM_ID_TEAM_HOME",
    away_team_id_col: str = "TEAM_ID_TEAM_AWAY",
    game_date_col: str = "GAME_DATE",
    season_year_col: str = "SEASON_YEAR",
    game_id_col: str = "GAME_ID",
    total_points_col: str = "TOTAL_POINTS",
    diff_from_line_col: str = "DIFF_FROM_LINE",
    home_player_cols: tuple[str, str, str],
    away_player_cols: tuple[str, str, str],
    out_prefix: str,
) -> pd.DataFrame:
    """
    Same computation as before, but the player-id columns are parameterized.
    This allows reuse for:
      - top scorers (non injured)
      - top injured players
      - any other top3 player list you have in the DF
    """
    df = df_games.copy()
    df = _ensure_datetime(df, game_date_col)

    # Check if DIFF_FROM_LINE exists, if not calculate it
    created_diff_from_line = False
    if diff_from_line_col not in df.columns:
        if total_points_col in df.columns and "TOTAL_OVER_UNDER_LINE" in df.columns:
            df[diff_from_line_col] = df[total_points_col] - df["TOTAL_OVER_UNDER_LINE"]
            created_diff_from_line = True
        else:
            raise ValueError(
                f"Column {diff_from_line_col} is missing and cannot be calculated. "
                f"Need both {total_points_col} and TOTAL_OVER_UNDER_LINE columns."
            )

    required = [
        home_team_id_col,
        away_team_id_col,
        game_date_col,
        season_year_col,
        game_id_col,
        total_points_col,
        diff_from_line_col,
        *home_player_cols,
        *away_player_cols,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    injured_index = _build_injured_games_index(injured_dict)

    # Build TEAM-game history (two rows per game: one per team)
    hist_home = df[
        [
            home_team_id_col,
            game_date_col,
            season_year_col,
            total_points_col,
            diff_from_line_col,
            game_id_col,
        ]
    ].copy()
    hist_home = hist_home.rename(columns={home_team_id_col: "TEAM_ID"})

    hist_away = df[
        [
            away_team_id_col,
            game_date_col,
            season_year_col,
            total_points_col,
            diff_from_line_col,
            game_id_col,
        ]
    ].copy()
    hist_away = hist_away.rename(columns={away_team_id_col: "TEAM_ID"})

    df_hist = pd.concat([hist_home, hist_away], ignore_index=True)
    df_hist = df_hist.rename(
        columns={
            game_date_col: "GAME_DATE",
            season_year_col: "SEASON_YEAR",
            total_points_col: "TOTAL_POINTS",
            diff_from_line_col: "DIFF_FROM_LINE",
            game_id_col: "GAME_ID",
        }
    )
    df_hist["GAME_DATE"] = pd.to_datetime(df_hist["GAME_DATE"], errors="coerce")
    df_hist["SEASON_YEAR"] = pd.to_numeric(
        df_hist["SEASON_YEAR"], errors="coerce"
    ).astype("Int64")

    @lru_cache(maxsize=250_000)
    def _cached_effect(
        team_id: int, season_year: int, date_ordinal: int, player_id: int
    ) -> tuple[float, float]:
        season_years = _infer_last_two_season_years_for_row(season_year)
        before_date = pd.Timestamp.fromordinal(date_ordinal)
        df_team_hist = _get_recent_history_df(
            df_hist,
            team_id=team_id,
            season_years=season_years,
            before_date=before_date,
        )
        injured_games_for_player = injured_index.get(team_id, {}).get(player_id, set())
        return _compute_player_presence_effect(df_team_hist, injured_games_for_player)

    def _out_col(side: str, i: int, metric: str) -> str:
        return f"{out_prefix}_{side}_P{i}_{metric}"

    n = len(df)
    home_tp = np.zeros((n, 3), dtype="float64")
    home_dfl = np.zeros((n, 3), dtype="float64")
    away_tp = np.zeros((n, 3), dtype="float64")
    away_dfl = np.zeros((n, 3), dtype="float64")

    # itertuples is faster than apply for 25k rows
    for i, row in enumerate(df.itertuples(index=False)):
        date = getattr(row, game_date_col)
        season_year = getattr(row, season_year_col)
        home_team = getattr(row, home_team_id_col)
        away_team = getattr(row, away_team_id_col)

        if (
            pd.isna(date)
            or pd.isna(season_year)
            or pd.isna(home_team)
            or pd.isna(away_team)
        ):
            continue

        date_ts = pd.Timestamp(date)
        date_ord = date_ts.toordinal()

        try:
            season_year_int = int(season_year)
            home_team_int = int(home_team)
            away_team_int = int(away_team)
        except Exception:
            continue

        for j, col in enumerate(home_player_cols):
            pid = getattr(row, col)
            if pd.isna(pid) or pid in (0, "0"):
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            tp_eff, dfl_eff = _cached_effect(
                home_team_int, season_year_int, date_ord, pid_int
            )
            home_tp[i, j] = tp_eff
            home_dfl[i, j] = dfl_eff

        for j, col in enumerate(away_player_cols):
            pid = getattr(row, col)
            if pd.isna(pid) or pid in (0, "0"):
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            tp_eff, dfl_eff = _cached_effect(
                away_team_int, season_year_int, date_ord, pid_int
            )
            away_tp[i, j] = tp_eff
            away_dfl[i, j] = dfl_eff

    for j in range(3):
        df[_out_col("HOME", j + 1, "TOTAL_POINTS")] = home_tp[:, j]
        df[_out_col("HOME", j + 1, "DIFF_FROM_LINE")] = home_dfl[:, j]
        df[_out_col("AWAY", j + 1, "TOTAL_POINTS")] = away_tp[:, j]
        df[_out_col("AWAY", j + 1, "DIFF_FROM_LINE")] = away_dfl[:, j]

    df[f"{out_prefix}_HOME_MEAN_TOTAL_POINTS"] = home_tp.mean(axis=1)
    df[f"{out_prefix}_HOME_MEAN_DIFF_FROM_LINE"] = home_dfl.mean(axis=1)
    df[f"{out_prefix}_AWAY_MEAN_TOTAL_POINTS"] = away_tp.mean(axis=1)
    df[f"{out_prefix}_AWAY_MEAN_DIFF_FROM_LINE"] = away_dfl.mean(axis=1)

    # Drop DIFF_FROM_LINE if we created it
    if created_diff_from_line and diff_from_line_col in df.columns:
        df = df.drop(columns=[diff_from_line_col])

    return df
