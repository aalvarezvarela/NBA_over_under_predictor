from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import get_main_book, total_line_col
from tqdm import tqdm


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
) -> tuple[float, float, int, int, int]:
    """
    Returns:
      (mean(TOTAL_POINTS|present) - mean(TOTAL_POINTS|injured),
       mean(DIFF_FROM_LINE|present) - mean(DIFF_FROM_LINE|injured),
       n_inj_games, n_present_games, n_total_games)
    """
    if df_team_hist.empty:
        return np.nan, np.nan, 0, 0, 0

    valid_mask = pd.to_numeric(df_team_hist["TOTAL_POINTS"], errors="coerce").notna() & (
        pd.to_numeric(df_team_hist["DIFF_FROM_LINE"], errors="coerce").notna()
    )
    df_team_hist = df_team_hist.loc[valid_mask]
    if df_team_hist.empty:
        return np.nan, np.nan, 0, 0, 0

    if not injured_games_for_player:
        n_total = int(len(df_team_hist))
        return np.nan, np.nan, 0, n_total, n_total

    game_ids = pd.to_numeric(df_team_hist["GAME_ID"], errors="coerce").astype("Int64")
    df_team_hist = df_team_hist.assign(_GAME_ID_INT=game_ids)

    inj_mask = df_team_hist["_GAME_ID_INT"].isin(list(injured_games_for_player))
    df_inj = df_team_hist.loc[inj_mask]
    df_present = df_team_hist.loc[~inj_mask]
    n_inj = int(len(df_inj))
    n_present = int(len(df_present))
    n_total = int(len(df_team_hist))

    if n_inj == 0 or n_present == 0:
        return np.nan, np.nan, n_inj, n_present, n_total

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
        return np.nan, np.nan, n_inj, n_present, n_total

    return float(tot_present - tot_inj), float(dfl_present - dfl_inj), n_inj, n_present, n_total


def _shrink_effect(
    raw_effect: float, n_inj_games: int, n_present_games: int, k: float
) -> float:
    """
    Empirical-Bayes style shrinkage toward zero.
    """
    if pd.isna(raw_effect):
        return np.nan
    n_eff = min(int(n_inj_games), int(n_present_games))
    if n_eff <= 0:
        return np.nan
    if k <= 0:
        return float(raw_effect)
    return float(raw_effect * (n_eff / (n_eff + k)))


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
    total_line_book: str | None = None,
    home_player_cols: tuple[str, ...],
    away_player_cols: tuple[str, ...],
    out_prefix: str,
    shrinkage_k: float = 10.0,
    include_per_player_columns: bool = False,
) -> pd.DataFrame:
    """
    Compute player absence impact features from past games only (< current game date).

    Changes vs previous version:
      - DIFF_FROM_LINE is always computed against TOTAL_LINE_<main book> selected by
        total_line_book (or configured main book from config).
      - Effects are shrunk toward zero with:
        eff_shrunk = eff_raw * n_eff/(n_eff + k), where n_eff=min(n_inj, n_present).
      - Sample size signals are produced (inj/present/total game counts).
      - Aggregate summaries include mean and max-abs effects plus count sums.
      - Per-player columns can be disabled via include_per_player_columns=False.
    """
    df = df_games.copy()
    df = _ensure_datetime(df, game_date_col)

    selected_total_line_col = total_line_col(total_line_book or get_main_book())
    if selected_total_line_col not in df.columns:
        raise ValueError(
            f"Missing required total line column {selected_total_line_col}. "
            "This function computes DIFF_FROM_LINE using the configured main book."
        )
    if total_points_col not in df.columns:
        raise ValueError(
            f"Missing required column {total_points_col}. "
            "Cannot compute DIFF_FROM_LINE history for injury effects."
        )

    # Always align DIFF_FROM_LINE computation to selected main total line.
    internal_diff_col = "__DIFF_FROM_MAIN_LINE_INTERNAL__"
    df[internal_diff_col] = pd.to_numeric(df[total_points_col], errors="coerce") - pd.to_numeric(
        df[selected_total_line_col], errors="coerce"
    )

    required = [
        home_team_id_col,
        away_team_id_col,
        game_date_col,
        season_year_col,
        game_id_col,
        total_points_col,
        selected_total_line_col,
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
            internal_diff_col,
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
            internal_diff_col,
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
            internal_diff_col: "DIFF_FROM_LINE",
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
    ) -> tuple[float, float, int, int, int]:
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
    n_home_players = len(home_player_cols)
    n_away_players = len(away_player_cols)

    home_tp = np.full((n, n_home_players), np.nan, dtype="float64")
    home_dfl = np.full((n, n_home_players), np.nan, dtype="float64")
    away_tp = np.full((n, n_away_players), np.nan, dtype="float64")
    away_dfl = np.full((n, n_away_players), np.nan, dtype="float64")
    home_n_inj = np.zeros((n, n_home_players), dtype="float64")
    home_n_present = np.zeros((n, n_home_players), dtype="float64")
    home_n_total = np.zeros((n, n_home_players), dtype="float64")
    away_n_inj = np.zeros((n, n_away_players), dtype="float64")
    away_n_present = np.zeros((n, n_away_players), dtype="float64")
    away_n_total = np.zeros((n, n_away_players), dtype="float64")

    # itertuples is faster than apply for 25k rows
    for i, row in enumerate(
        tqdm(
            df.itertuples(index=False), total=len(df), desc="Computing absence effects"
        )
    ):
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

        seen_home_pids: set[int] = set()
        for j, col in enumerate(home_player_cols):
            pid = getattr(row, col)
            if pd.isna(pid) or pid in (0, "0"):
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if pid_int in seen_home_pids:
                continue
            seen_home_pids.add(pid_int)
            tp_raw, dfl_raw, n_inj, n_present, n_total = _cached_effect(
                home_team_int, season_year_int, date_ord, pid_int
            )
            home_tp[i, j] = _shrink_effect(tp_raw, n_inj, n_present, shrinkage_k)
            home_dfl[i, j] = _shrink_effect(dfl_raw, n_inj, n_present, shrinkage_k)
            home_n_inj[i, j] = n_inj
            home_n_present[i, j] = n_present
            home_n_total[i, j] = n_total

        seen_away_pids: set[int] = set()
        for j, col in enumerate(away_player_cols):
            pid = getattr(row, col)
            if pd.isna(pid) or pid in (0, "0"):
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if pid_int in seen_away_pids:
                continue
            seen_away_pids.add(pid_int)
            tp_raw, dfl_raw, n_inj, n_present, n_total = _cached_effect(
                away_team_int, season_year_int, date_ord, pid_int
            )
            away_tp[i, j] = _shrink_effect(tp_raw, n_inj, n_present, shrinkage_k)
            away_dfl[i, j] = _shrink_effect(dfl_raw, n_inj, n_present, shrinkage_k)
            away_n_inj[i, j] = n_inj
            away_n_present[i, j] = n_present
            away_n_total[i, j] = n_total

    if include_per_player_columns:
        for j in range(n_home_players):
            df[_out_col("HOME", j + 1, "TOTAL_POINTS")] = home_tp[:, j]
            df[_out_col("HOME", j + 1, diff_from_line_col)] = home_dfl[:, j]
            df[_out_col("HOME", j + 1, "N_INJ_GAMES")] = home_n_inj[:, j]
            df[_out_col("HOME", j + 1, "N_PRESENT_GAMES")] = home_n_present[:, j]
            df[_out_col("HOME", j + 1, "N_TOTAL_GAMES")] = home_n_total[:, j]

        for j in range(n_away_players):
            df[_out_col("AWAY", j + 1, "TOTAL_POINTS")] = away_tp[:, j]
            df[_out_col("AWAY", j + 1, diff_from_line_col)] = away_dfl[:, j]
            df[_out_col("AWAY", j + 1, "N_INJ_GAMES")] = away_n_inj[:, j]
            df[_out_col("AWAY", j + 1, "N_PRESENT_GAMES")] = away_n_present[:, j]
            df[_out_col("AWAY", j + 1, "N_TOTAL_GAMES")] = away_n_total[:, j]

    def _nanmean_axis1(arr: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(arr)
        counts = mask.sum(axis=1)
        sums = np.nansum(arr, axis=1)
        out = np.full(arr.shape[0], np.nan, dtype="float64")
        valid = counts > 0
        out[valid] = sums[valid] / counts[valid]
        return out

    def _nanmaxabs_axis1(arr: np.ndarray) -> np.ndarray:
        abs_arr = np.abs(arr)
        out = np.full(arr.shape[0], np.nan, dtype="float64")
        valid = ~np.isnan(abs_arr).all(axis=1)
        if valid.any():
            out[valid] = np.nanmax(abs_arr[valid], axis=1)
        return out

    df[f"{out_prefix}_HOME_MEAN_TOTAL_POINTS"] = _nanmean_axis1(home_tp)
    df[f"{out_prefix}_AWAY_MEAN_TOTAL_POINTS"] = _nanmean_axis1(away_tp)
    df[f"{out_prefix}_HOME_MEAN_{diff_from_line_col}"] = _nanmean_axis1(home_dfl)
    df[f"{out_prefix}_AWAY_MEAN_{diff_from_line_col}"] = _nanmean_axis1(away_dfl)
    df[f"{out_prefix}_HOME_MAX_ABS_TOTAL_POINTS"] = _nanmaxabs_axis1(home_tp)
    df[f"{out_prefix}_AWAY_MAX_ABS_TOTAL_POINTS"] = _nanmaxabs_axis1(away_tp)
    df[f"{out_prefix}_HOME_MAX_ABS_{diff_from_line_col}"] = _nanmaxabs_axis1(home_dfl)
    df[f"{out_prefix}_AWAY_MAX_ABS_{diff_from_line_col}"] = _nanmaxabs_axis1(away_dfl)
    df[f"{out_prefix}_HOME_SUM_N_INJ_GAMES"] = home_n_inj.sum(axis=1)
    df[f"{out_prefix}_AWAY_SUM_N_INJ_GAMES"] = away_n_inj.sum(axis=1)
    df[f"{out_prefix}_HOME_SUM_N_PRESENT_GAMES"] = home_n_present.sum(axis=1)
    df[f"{out_prefix}_AWAY_SUM_N_PRESENT_GAMES"] = away_n_present.sum(axis=1)
    df[f"{out_prefix}_HOME_SUM_N_TOTAL_GAMES"] = home_n_total.sum(axis=1)
    df[f"{out_prefix}_AWAY_SUM_N_TOTAL_GAMES"] = away_n_total.sum(axis=1)
    df[f"{out_prefix}_HOME_N_PLAYERS_WITH_EFFECT"] = (
        (home_n_inj > 0) & (home_n_present > 0)
    ).sum(axis=1)
    df[f"{out_prefix}_AWAY_N_PLAYERS_WITH_EFFECT"] = (
        (away_n_inj > 0) & (away_n_present > 0)
    ).sum(axis=1)
    df[f"{out_prefix}_HOME_HAS_PLAYER_EFFECT"] = (
        df[f"{out_prefix}_HOME_N_PLAYERS_WITH_EFFECT"] > 0
    ).astype(int)
    df[f"{out_prefix}_AWAY_HAS_PLAYER_EFFECT"] = (
        df[f"{out_prefix}_AWAY_N_PLAYERS_WITH_EFFECT"] > 0
    ).astype(int)

    # Cleanup internal helper column.
    if internal_diff_col in df.columns:
        df = df.drop(columns=[internal_diff_col])

    return df
