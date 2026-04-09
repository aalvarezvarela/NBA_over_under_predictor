from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.modeling.scorers import over_under_betting_accuracy_error_line

try:
    from meta_learner_baselines import (
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
        add_default_meta_learner_baselines,
    )
except ModuleNotFoundError:
    from lab.meta_learner.meta_learner_baselines import (
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
        add_default_meta_learner_baselines,
    )

DATE_COL = "GAME_DATE"
GAME_ID_COL = "GAME_ID"
SEASON_COL = "SEASON_YEAR"
TARGET_ERROR_COL = "LINE_ERROR"
DEFAULT_ROLLING_WINDOW_DAYS = 5
DEFAULT_LINE_BUCKET_EDGES = [205.0, 215.0, 225.0, 235.0, 245.0]
DEFAULT_CONFIDENCE_BUCKET_EDGES = [1.0, 2.0, 3.0, 5.0]


@dataclass(frozen=True)
class MetaLearnerFeatureBuildResult:
    dataframe: pd.DataFrame
    line_col: str
    total_model_cols: list[str]
    line_error_model_cols: list[str]
    error_cols: list[str]
    vote_feature_cols: list[str]
    reliability_feature_cols: list[str]
    baseline_cols: list[str]


def load_meta_learner_dataframe(
    csv_path: str | Path,
    *,
    date_col: str = DATE_COL,
    game_id_col: str = GAME_ID_COL,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    dtype_dict = {col: str for col in header_cols if "ID" in col.upper()}
    df = pd.read_csv(csv_path, dtype=dtype_dict)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    sort_cols = [column for column in [date_col, game_id_col] if column in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def resolve_prediction_column_groups(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    total_model_cols = [
        column for column in df.columns if column.startswith("PRED_TOTAL_POINTS_")
    ]
    line_error_model_cols = [
        column for column in df.columns if column.startswith("PRED_LINE_ERROR_")
    ]
    if not total_model_cols and not line_error_model_cols:
        raise KeyError("No meta-learner prediction columns were found in the dataframe.")
    return total_model_cols, line_error_model_cols


def resolve_line_column(
    df: pd.DataFrame,
    *,
    preferred_line_col: str | None = None,
) -> str:
    candidates = [
        column
        for column in [preferred_line_col, total_line_col(), "TOTAL_LINE_bet365"]
        if column
    ]
    for column in candidates:
        if column in df.columns:
            return column

    fallback_columns = [column for column in df.columns if column.startswith("TOTAL_LINE_")]
    if fallback_columns:
        return fallback_columns[0]

    raise KeyError("Could not resolve a total-line column from the dataframe.")


def build_error_space_prediction_frame(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str],
    line_error_model_cols: list[str],
    line_col: str,
) -> pd.DataFrame:
    line_values = pd.to_numeric(df[line_col], errors="coerce")
    converted: dict[str, pd.Series] = {}

    for column in total_model_cols:
        converted[f"{column}__ERR"] = pd.to_numeric(df[column], errors="coerce") - line_values

    for column in line_error_model_cols:
        converted[f"{column}__ERR"] = pd.to_numeric(df[column], errors="coerce")

    return pd.DataFrame(converted, index=df.index)


def add_error_space_prediction_columns(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str],
    line_error_model_cols: list[str],
    line_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    error_space_df = build_error_space_prediction_frame(
        output,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    output = pd.concat([output, error_space_df], axis=1)
    return output, error_space_df.columns.tolist()


def _shannon_entropy_from_counts(count_matrix: np.ndarray) -> np.ndarray:
    totals = count_matrix.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        count_matrix,
        totals,
        out=np.zeros_like(count_matrix, dtype=float),
        where=totals > 0,
    )
    log_probabilities = np.zeros_like(probabilities, dtype=float)
    positive_mask = probabilities > 0.0
    log_probabilities[positive_mask] = np.log2(probabilities[positive_mask])
    return -np.sum(probabilities * log_probabilities, axis=1)


def add_vote_summary_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    error_values = output[error_cols].apply(pd.to_numeric, errors="coerce")
    error_array = error_values.to_numpy(dtype=float)
    sign_array = np.sign(error_array)
    valid_mask = np.isfinite(error_array)

    valid_count = valid_mask.sum(axis=1)
    over_count = (sign_array > 0).sum(axis=1)
    under_count = (sign_array < 0).sum(axis=1)
    push_count = ((sign_array == 0) & valid_mask).sum(axis=1)

    vote_feature_cols = [
        "META_VALID_VOTE_COUNT",
        "META_OVER_VOTE_COUNT",
        "META_UNDER_VOTE_COUNT",
        "META_PUSH_VOTE_COUNT",
        "META_VOTE_PROP_OVER",
        "META_VOTE_PROP_UNDER",
        "META_VOTE_PROP_PUSH",
        "META_VOTE_MEAN_ERR",
        "META_VOTE_MEDIAN_ERR",
        "META_VOTE_STD_ERR",
        "META_VOTE_MIN_ERR",
        "META_VOTE_MAX_ERR",
        "META_VOTE_RANGE_ERR",
        "META_VOTE_ENTROPY",
    ]

    output["META_VALID_VOTE_COUNT"] = valid_count
    output["META_OVER_VOTE_COUNT"] = over_count
    output["META_UNDER_VOTE_COUNT"] = under_count
    output["META_PUSH_VOTE_COUNT"] = push_count
    output["META_VOTE_PROP_OVER"] = np.divide(
        over_count,
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )
    output["META_VOTE_PROP_UNDER"] = np.divide(
        under_count,
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )
    output["META_VOTE_PROP_PUSH"] = np.divide(
        push_count,
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )
    output["META_VOTE_MEAN_ERR"] = np.nanmean(error_array, axis=1)
    output["META_VOTE_MEDIAN_ERR"] = np.nanmedian(error_array, axis=1)
    output["META_VOTE_STD_ERR"] = np.nanstd(error_array, axis=1)
    output["META_VOTE_MIN_ERR"] = np.nanmin(error_array, axis=1)
    output["META_VOTE_MAX_ERR"] = np.nanmax(error_array, axis=1)
    output["META_VOTE_RANGE_ERR"] = output["META_VOTE_MAX_ERR"] - output["META_VOTE_MIN_ERR"]
    output["META_VOTE_ENTROPY"] = _shannon_entropy_from_counts(
        np.column_stack([over_count, under_count, push_count])
    )

    no_vote_mask = valid_count == 0
    output.loc[no_vote_mask, vote_feature_cols] = np.nan
    return output, vote_feature_cols


def _bucket_codes(values: np.ndarray, inner_edges: list[float]) -> np.ndarray:
    bins = [-np.inf, *inner_edges, np.inf]
    bucket_codes = pd.cut(
        pd.Series(values, copy=False),
        bins=bins,
        labels=False,
        include_lowest=True,
    )
    return bucket_codes.to_numpy(dtype=float)


def _date_group_bounds(dates: pd.Series) -> list[tuple[int, int]]:
    date_days = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    if len(date_days) == 0:
        return []
    group_starts = np.flatnonzero(np.r_[True, date_days[1:] != date_days[:-1]])
    group_ends = np.r_[group_starts[1:], len(date_days)]
    return list(zip(group_starts, group_ends, strict=True))


def _compute_accuracy(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan

    true_side = np.sign(y_true_error[valid])
    pred_side = np.sign(y_pred_error[valid])
    eligible = (true_side != 0) & (pred_side != 0)
    if not np.any(eligible):
        return np.nan

    return float(np.mean(true_side[eligible] == pred_side[eligible]))


def _compute_mae(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(y_pred_error[valid] - y_true_error[valid])))


def _compute_bias(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan
    return float(np.mean(y_pred_error[valid] - y_true_error[valid]))


def add_rolling_reliability_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
    line_col: str,
    date_col: str = DATE_COL,
    target_error_col: str = TARGET_ERROR_COL,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    line_bucket_edges: list[float] | None = None,
    confidence_bucket_edges: list[float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if rolling_window_days <= 0:
        raise ValueError("rolling_window_days must be greater than zero.")
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")
    if target_error_col not in df.columns:
        raise KeyError(f"Missing required target error column: {target_error_col}")
    if line_col not in df.columns:
        raise KeyError(f"Missing required line column: {line_col}")

    output = df.copy()
    line_bucket_edges = line_bucket_edges or list(DEFAULT_LINE_BUCKET_EDGES)
    confidence_bucket_edges = confidence_bucket_edges or list(
        DEFAULT_CONFIDENCE_BUCKET_EDGES
    )

    y_true = pd.to_numeric(output[target_error_col], errors="coerce").to_numpy(dtype=float)
    line_values = pd.to_numeric(output[line_col], errors="coerce").to_numpy(dtype=float)
    line_bucket_codes = _bucket_codes(line_values, line_bucket_edges)
    output["META_TOTAL_LINE_BUCKET"] = line_bucket_codes

    date_groups = _date_group_bounds(output[date_col])
    reliability_feature_cols: list[str] = ["META_TOTAL_LINE_BUCKET"]

    for error_col in error_cols:
        y_pred = pd.to_numeric(output[error_col], errors="coerce").to_numpy(dtype=float)
        conf_bucket_col = f"{error_col}_CONF_BUCKET"
        conf_bucket_codes = _bucket_codes(np.abs(y_pred), confidence_bucket_edges)
        output[conf_bucket_col] = conf_bucket_codes
        reliability_feature_cols.append(conf_bucket_col)

        feature_arrays = {
            f"{error_col}_ROLLING_HISTORY_N_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_MAE_{rolling_window_days}D": np.full(len(output), np.nan),
            f"{error_col}_ROLLING_BETTING_ACC_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_BIAS_{rolling_window_days}D": np.full(len(output), np.nan),
            f"{error_col}_ROLLING_LINE_BUCKET_N_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_LINE_BUCKET_MAE_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_LINE_BUCKET_ACC_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_CONF_BUCKET_N_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_CONF_BUCKET_MAE_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_CONF_BUCKET_ACC_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
        }

        for group_idx, (start, end) in enumerate(date_groups):
            if group_idx == 0:
                history_slice = slice(0, 0)
            else:
                history_group_start_idx = max(0, group_idx - rolling_window_days)
                history_start = date_groups[history_group_start_idx][0]
                history_slice = slice(history_start, start)

            history_true = y_true[history_slice]
            history_pred = y_pred[history_slice]
            history_line_buckets = line_bucket_codes[history_slice]
            history_conf_buckets = conf_bucket_codes[history_slice]
            history_valid_mask = np.isfinite(history_true) & np.isfinite(history_pred)

            feature_arrays[f"{error_col}_ROLLING_HISTORY_N_{rolling_window_days}D"][
                start:end
            ] = float(history_valid_mask.sum())
            feature_arrays[f"{error_col}_ROLLING_MAE_{rolling_window_days}D"][start:end] = (
                _compute_mae(history_true, history_pred)
            )
            feature_arrays[
                f"{error_col}_ROLLING_BETTING_ACC_{rolling_window_days}D"
            ][start:end] = _compute_accuracy(history_true, history_pred)
            feature_arrays[f"{error_col}_ROLLING_BIAS_{rolling_window_days}D"][start:end] = (
                _compute_bias(history_true, history_pred)
            )

            for row_idx in range(start, end):
                current_line_bucket = line_bucket_codes[row_idx]
                if np.isfinite(current_line_bucket):
                    line_bucket_mask = history_valid_mask & (
                        history_line_buckets == current_line_bucket
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_LINE_BUCKET_N_{rolling_window_days}D"
                    ][row_idx] = float(line_bucket_mask.sum())
                    feature_arrays[
                        f"{error_col}_ROLLING_LINE_BUCKET_MAE_{rolling_window_days}D"
                    ][row_idx] = _compute_mae(
                        history_true[line_bucket_mask],
                        history_pred[line_bucket_mask],
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_LINE_BUCKET_ACC_{rolling_window_days}D"
                    ][row_idx] = _compute_accuracy(
                        history_true[line_bucket_mask],
                        history_pred[line_bucket_mask],
                    )

                current_conf_bucket = conf_bucket_codes[row_idx]
                if np.isfinite(current_conf_bucket):
                    conf_bucket_mask = history_valid_mask & (
                        history_conf_buckets == current_conf_bucket
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_N_{rolling_window_days}D"
                    ][row_idx] = float(conf_bucket_mask.sum())
                    feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_MAE_{rolling_window_days}D"
                    ][row_idx] = _compute_mae(
                        history_true[conf_bucket_mask],
                        history_pred[conf_bucket_mask],
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_ACC_{rolling_window_days}D"
                    ][row_idx] = _compute_accuracy(
                        history_true[conf_bucket_mask],
                        history_pred[conf_bucket_mask],
                    )

        for column_name, values in feature_arrays.items():
            output[column_name] = values
            reliability_feature_cols.append(column_name)

    return output, reliability_feature_cols


def summarize_prediction(
    y_true_error: pd.Series,
    y_pred_error: pd.Series,
    *,
    label: str,
) -> dict[str, float | int | str]:
    y_true = pd.to_numeric(y_true_error, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(y_pred_error, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    true_side = np.sign(y_true)
    pred_side = np.sign(y_pred)
    eligible = valid & (true_side != 0) & (pred_side != 0)
    actual_non_push = valid & (true_side != 0)
    residual = y_pred - y_true

    accuracy = over_under_betting_accuracy_error_line(
        y_true_error=y_true[valid],
        y_pred_error=y_pred[valid],
    )
    coverage = np.nan
    if actual_non_push.sum() > 0:
        coverage = 100.0 * eligible.sum() / actual_non_push.sum()

    return {
        "model": label,
        "accuracy_pct": 100.0 * accuracy,
        "eligible_games": int(eligible.sum()),
        "coverage_pct_non_push": float(coverage),
        "pred_push_rate_pct": 100.0 * np.mean(pred_side[valid] == 0) if np.any(valid) else np.nan,
        "mean_abs_predicted_edge": float(np.nanmean(np.abs(y_pred[valid]))) if np.any(valid) else np.nan,
        "mean_predicted_edge": float(np.nanmean(y_pred[valid])) if np.any(valid) else np.nan,
        "mae_vs_true_error": float(np.nanmean(np.abs(residual[valid]))) if np.any(valid) else np.nan,
        "rmse_vs_true_error": float(np.sqrt(np.nanmean(np.square(residual[valid])))) if np.any(valid) else np.nan,
        "bias_vs_true_error": float(np.nanmean(residual[valid])) if np.any(valid) else np.nan,
    }


def summarize_prediction_columns(
    df: pd.DataFrame,
    *,
    prediction_cols: list[str],
    target_error_col: str = TARGET_ERROR_COL,
) -> pd.DataFrame:
    rows = [
        summarize_prediction(df[target_error_col], df[column], label=column)
        for column in prediction_cols
    ]
    return pd.DataFrame(rows).sort_values("accuracy_pct", ascending=False).reset_index(
        drop=True
    )


def build_meta_learner_feature_frame(
    df: pd.DataFrame,
    *,
    preferred_line_col: str | None = None,
    date_col: str = DATE_COL,
    game_id_col: str = GAME_ID_COL,
    target_error_col: str = TARGET_ERROR_COL,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    line_bucket_edges: list[float] | None = None,
    confidence_bucket_edges: list[float] | None = None,
) -> MetaLearnerFeatureBuildResult:
    total_model_cols, line_error_model_cols = resolve_prediction_column_groups(df)
    line_col = resolve_line_column(df, preferred_line_col=preferred_line_col)

    working = df.copy()
    if date_col in working.columns:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce").dt.normalize()
    sort_cols = [column for column in [date_col, game_id_col] if column in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    working, error_cols = add_error_space_prediction_columns(
        working,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    working, vote_feature_cols = add_vote_summary_features(working, error_cols=error_cols)
    working, reliability_feature_cols = add_rolling_reliability_features(
        working,
        error_cols=error_cols,
        line_col=line_col,
        date_col=date_col,
        target_error_col=target_error_col,
        rolling_window_days=rolling_window_days,
        line_bucket_edges=line_bucket_edges,
        confidence_bucket_edges=confidence_bucket_edges,
    )
    working = add_default_meta_learner_baselines(
        working,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )

    baseline_cols = [
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
    ]

    return MetaLearnerFeatureBuildResult(
        dataframe=working,
        line_col=line_col,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        error_cols=error_cols,
        vote_feature_cols=vote_feature_cols,
        reliability_feature_cols=reliability_feature_cols,
        baseline_cols=baseline_cols,
    )
