"""The "trust the bookmaker's line" naive baseline.

No such baseline exists anywhere else in the repo (confirmed by a full-repo
search this session): nba_ou.prediction.baseline_predictions only has a
random +/-1 point baseline and a historical-mean-TOTAL_POINTS baseline --
neither treats the betting line itself as the forecast.

Formula: for a TOTAL_POINTS run, PRED = df[baseline_line_col]; for a
LINE_ERROR run, PRED = 0 for every row. Both express the same underlying
claim ("the bookmaker line is unbiased"), so compute_baseline_metrics always
scores in TOTAL_POINTS space regardless of which target the run trains on --
this keeps baseline numbers numerically comparable across both target
families and directly comparable to TrainingMetrics.final_test_mae/cv_mae for
TOTAL_POINTS runs.

ou_accuracy is always reported as None (serialized as NaN), never 0.0:
predicted edge (pred - line) is exactly 0 for every row, so every row is a
"push" under the existing scorers' push-exclusion convention in
nba_ou.modeling.scorers (over_under_betting_accuracy_total_points /
over_under_betting_accuracy_error_line both return 0.0 when no non-push rows
remain). Reporting 0.0 here would misleadingly read as "the baseline is
always wrong"; None/NaN correctly reads as "no defined bet exists for this
baseline."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaselineMetrics(BaseModel):
    line_col: str
    n_games: int
    mae: float
    rmse: float
    r2: float
    ou_accuracy: float | None = None


def compute_baseline_metrics(
    *,
    y_true_total_points: pd.Series,
    baseline_line: pd.Series,
    line_col: str,
) -> BaselineMetrics:
    y_true = pd.to_numeric(y_true_total_points, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(baseline_line, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError(
            f"No valid (finite) rows to score the baseline against column {line_col!r}."
        )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    return BaselineMetrics(
        line_col=line_col,
        n_games=int(valid.sum()),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        ou_accuracy=None,
    )


def compute_line_error_bias(
    df: pd.DataFrame, *, baseline_line_col: str
) -> float:
    """Mean signed line error (actual total minus line) over the given rows.

    Must be computed on development rows only and then applied to the holdout,
    otherwise the "bias-corrected" baseline peeks at the test period.
    """
    actual = pd.to_numeric(df["TOTAL_POINTS"], errors="coerce")
    line = pd.to_numeric(df[baseline_line_col], errors="coerce")
    bias = (actual - line).mean()
    if not np.isfinite(bias):
        raise ValueError("Could not compute a finite line-error bias.")
    return float(bias)


def compute_bias_corrected_baseline_metrics(
    df: pd.DataFrame,
    *,
    baseline_line_col: str,
    bias: float,
) -> BaselineMetrics:
    """A deliberately harder null than "trust the line".

    Predicts ``line + bias``, where ``bias`` is the average amount by which
    games historically exceeded the line. If a model cannot beat this, it has
    only rediscovered a league-wide over/under drift rather than learned
    anything game-specific. Unlike the pure line baseline this one has a
    non-zero edge on every row, so it does place bets and has a defined win
    rate.
    """
    actual = pd.to_numeric(df["TOTAL_POINTS"], errors="coerce")
    line = pd.to_numeric(df[baseline_line_col], errors="coerce")
    return compute_baseline_metrics(
        y_true_total_points=actual,
        baseline_line=line + bias,
        line_col=f"{baseline_line_col}+bias({bias:+.3f})",
    )


def compute_baseline_metrics_for_rows(
    df_full: pd.DataFrame,
    idx: np.ndarray,
    *,
    baseline_line_col: str,
) -> BaselineMetrics:
    rows = df_full.iloc[idx]
    return compute_baseline_metrics(
        y_true_total_points=rows["TOTAL_POINTS"],
        baseline_line=rows[baseline_line_col],
        line_col=baseline_line_col,
    )


def compute_baseline_metrics_across_folds(
    df_dev_full: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    baseline_line_col: str,
) -> tuple[pd.DataFrame, BaselineMetrics]:
    """Compute the baseline on each CV fold's *validation* rows (never train rows).

    Returns a per-fold table (to sit next to the Optuna trial summary's
    per-fold MAE/RMSE) and one aggregate BaselineMetrics: the aggregate MAE is
    the mean of fold MAEs, matching how the Optuna objective itself aggregates
    mean_mae across folds -- so the two numbers are directly comparable.
    """
    fold_rows: list[dict] = []
    for fold_num, (_, val_idx) in enumerate(splits, start=1):
        fold_metrics = compute_baseline_metrics_for_rows(
            df_dev_full, val_idx, baseline_line_col=baseline_line_col
        )
        fold_rows.append(
            {
                "fold": fold_num,
                "n_games": fold_metrics.n_games,
                "mae": fold_metrics.mae,
                "rmse": fold_metrics.rmse,
                "r2": fold_metrics.r2,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    aggregate = BaselineMetrics(
        line_col=baseline_line_col,
        n_games=int(fold_df["n_games"].sum()),
        mae=float(fold_df["mae"].mean()),
        rmse=float(fold_df["rmse"].mean()),
        r2=float(fold_df["r2"].mean()),
        ou_accuracy=None,
    )
    return fold_df, aggregate
