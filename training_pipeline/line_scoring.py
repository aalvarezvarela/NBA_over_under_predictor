"""Re-scoring predictions against lines other than the one bets settled into.

Why this exists: everything else in the pipeline scores against the CLOSING
total line. That is the right bar for "does this model know something" -- the
close is the market's final, sharpest price -- but it is not a line anyone can
actually bet, because by definition it is the last price quoted before tip-off.

Measured on this repo's own training data, the line moves 2.54 points on
average and moves at all in 93% of games. The default bet trigger is 2.0
points. So the movement is larger than the threshold that decides whether to
bet, meaning an edge measured against the close can correspond to a
substantively different bet against the open. Reporting both separates two
claims that are easy to conflate:

  - beating the close  => the model has information the market lacked
  - beating the open   => that information was capturable at a real price

This module is informational only. Nothing here feeds training, tuning, trial
selection, or the headline metrics; it just re-scores predictions that already
exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training_pipeline.betting import evaluate_alternative_lines
from training_pipeline.config import ExperimentConfig, TargetFamily


def predicted_total_points(
    y_pred: np.ndarray,
    *,
    target_line: np.ndarray,
    target_family: TargetFamily,
) -> np.ndarray:
    """Model predictions expressed in TOTAL_POINTS space.

    An edge is only meaningful relative to the line it came from, so comparing
    lines requires first putting the prediction back into absolute points. For
    a TOTAL_POINTS model that is the prediction itself; a LINE_ERROR model
    predicts a quantity relative to its own line, so the line is added back.
    """
    if target_family == TargetFamily.LINE_ERROR:
        return np.asarray(target_line, dtype=float) + np.asarray(y_pred, dtype=float)
    return np.asarray(y_pred, dtype=float)


def collect_comparison_lines(
    df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    target_line_col: str,
    positions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Line columns to compare, keyed by column name.

    The target line always comes first so it acts as the reference point for
    ``mean_abs_move_vs_first``. Configured columns missing from ``df`` are
    skipped rather than raising: line availability genuinely varies between CSV
    snapshots, and this is a diagnostic, so it must never be able to fail a run
    that would otherwise have succeeded.

    ``positions`` positionally selects rows (the walk-forward predicts a subset
    of the evaluation frame, in day order rather than row order).
    """
    lines: dict[str, np.ndarray] = {}
    for column in (target_line_col, *config.betting.comparison_line_cols):
        if column in lines or column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        lines[column] = values if positions is None else values[positions]
    return lines


def build_line_comparison(
    *,
    y_pred: np.ndarray,
    target_line: np.ndarray,
    actual_total: np.ndarray,
    lines: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> pd.DataFrame | None:
    """The alternative-line table, or None when there is nothing to compare.

    Returns None rather than a one-row frame when only the target line is
    available: a comparison of a line against itself carries no information and
    would only add a file to every run directory.
    """
    if len(lines) < 2:
        return None

    return evaluate_alternative_lines(
        predicted_total_points=predicted_total_points(
            y_pred, target_line=target_line, target_family=config.family
        ),
        actual_total=actual_total,
        lines=lines,
        min_edge=config.betting.primary_edge_threshold,
        flat_decimal_odds=config.betting.flat_decimal_odds,
    )
