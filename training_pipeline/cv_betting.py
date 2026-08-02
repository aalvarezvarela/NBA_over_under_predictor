"""Betting metrics across the cross-validation folds, not just the holdout.

The motivation is statistical power. At -110 the break-even win rate is 52.38%,
and a Wilson interval around a genuinely good 55% win rate does not clear that
bar until well over a thousand bets. A 5% holdout of this dataset yields ~290
candidate games and ~115 actual bets, which cannot distinguish a real edge from
a lucky one -- any ranking decided on holdout ROI alone at that volume is close
to a coin flip.

The CV folds already exist and already cover far more games: 12 folds of ~50
validation games is ~600, roughly 5x the holdout. Scoring them for profit costs
one extra fit per fold at the already-chosen hyperparameters -- about the same
as a single Optuna trial -- and turns the split that actually drives decisions
into the split with the most evidence behind it.

The honest caveat, which is why this does not replace the holdout: these folds
selected the hyperparameters, so their betting metrics are optimistically
biased. Use them to COMPARE configurations (the bias applies roughly equally to
each) and the holdout to ESTIMATE out-of-sample performance. The gap between
the two is itself informative -- a large one is the signature of selection
overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training_pipeline.betting import (
    BettingMetrics,
    betting_threshold_sweep,
    evaluate_betting,
)
from training_pipeline.config import ExperimentConfig, TargetFamily
from training_pipeline.line_scoring import (
    build_line_comparison,
    collect_comparison_lines,
)
from training_pipeline.tuning import fit_final_model


@dataclass
class CrossValidationBettingResult:
    """Profit metrics over the pooled CV validation rows."""

    n_folds: int
    #: Pooled validation rows. Equals n_unique_games unless the fold layout
    #: overlaps (step_games_between_tests < test_games), in which case some
    #: games are counted more than once -- which is why both are reported.
    n_games: int
    n_unique_games: int
    mae: float
    rmse: float
    #: Mean of the per-fold MAEs, matching how the Optuna objective aggregates
    #: mean_mae -- so this is directly comparable to the tuned cv_mae, while
    #: ``mae`` above is pooled over games and weights larger folds more.
    mean_fold_mae: float
    betting_primary: BettingMetrics
    betting_sweep: pd.DataFrame
    #: One row per fold: size, error and profit. Read this before the pooled
    #: number -- an ROI carried by one fold is a different claim from one that
    #: held across twelve.
    fold_metrics: pd.DataFrame
    #: One row per pooled validation game.
    predictions: pd.DataFrame
    line_comparison: pd.DataFrame | None
    n_profitable_folds: int
    random_state: int

    def summary(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_games": self.n_games,
            "n_unique_games": self.n_unique_games,
            "mae": self.mae,
            "rmse": self.rmse,
            "mean_fold_mae": self.mean_fold_mae,
            "roi": self.betting_primary.roi,
            "n_bets": self.betting_primary.n_bets,
            "bet_rate": self.betting_primary.bet_rate,
            "win_rate": self.betting_primary.win_rate,
            "win_rate_ci_low": self.betting_primary.win_rate_ci_low,
            "win_rate_ci_high": self.betting_primary.win_rate_ci_high,
            "break_even_rate": self.betting_primary.break_even_rate,
            "is_significant": self.betting_primary.is_significant,
            "profit_units": self.betting_primary.profit_units,
            "n_profitable_folds": self.n_profitable_folds,
            "random_state": self.random_state,
        }


def evaluate_cv_betting(
    config: ExperimentConfig,
    *,
    df_dev: pd.DataFrame,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    dates_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    n_estimators: int,
    target_line_col: str,
    sample_weight_lambda: float | None = None,
    random_state: int | None = None,
) -> CrossValidationBettingResult:
    """Refit at the chosen hyperparameters on each fold and score for profit.

    One fit per fold, on that fold's training rows only, predicting only its
    validation rows -- the same time-aware discipline the tuning objective
    used, so no fold ever sees its own future.
    """
    resolved_random_state = (
        config.random_state if random_state is None else random_state
    )
    is_line_error = config.target_family == TargetFamily.LINE_ERROR

    target_line_all = pd.to_numeric(
        df_dev[target_line_col], errors="coerce"
    ).to_numpy(dtype=float)
    actual_total_all = pd.to_numeric(
        df_dev["TOTAL_POINTS"], errors="coerce"
    ).to_numpy(dtype=float)

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_num, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = fit_final_model(
            X_dev=X_dev.iloc[train_idx],
            y_dev=y_dev.iloc[train_idx],
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev.iloc[train_idx],
            sample_weight_lambda=sample_weight_lambda,
            random_state=resolved_random_state,
        )

        y_true = pd.to_numeric(y_dev.iloc[valid_idx], errors="coerce").to_numpy(
            dtype=float
        )
        y_pred = np.asarray(model.predict(X_dev.iloc[valid_idx]), dtype=float)

        fold_line = target_line_all[valid_idx]
        fold_actual = actual_total_all[valid_idx]
        fold_edge = y_pred if is_line_error else y_pred - fold_line

        fold_betting = evaluate_betting(
            predicted_edge=fold_edge,
            actual_total=fold_actual,
            line=fold_line,
            min_edge=config.betting.primary_edge_threshold,
            flat_decimal_odds=config.betting.flat_decimal_odds,
        )

        fold_rows.append(
            {
                "fold": fold_num,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "valid_start": pd.Timestamp(dates_dev.iloc[valid_idx].min()),
                "valid_end": pd.Timestamp(dates_dev.iloc[valid_idx].max()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                # The line's own error on the same rows, so each fold shows
                # whether the model beat the market there or merely tracked it.
                "line_mae": float(np.mean(np.abs(fold_actual - fold_line))),
                "n_bets": fold_betting.n_bets,
                "win_rate": fold_betting.win_rate,
                "roi": fold_betting.roi,
                "profit_units": fold_betting.profit_units,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_num,
                    "row_in_dev": valid_idx,
                    config.data.date_col: dates_dev.iloc[valid_idx].to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "target_line": fold_line,
                    "TOTAL_POINTS": fold_actual,
                    "predicted_edge": fold_edge,
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    pooled_edge = predictions["predicted_edge"].to_numpy(dtype=float)
    pooled_line = predictions["target_line"].to_numpy(dtype=float)
    pooled_actual = predictions["TOTAL_POINTS"].to_numpy(dtype=float)

    betting_kwargs: dict[str, Any] = {
        "actual_total": pooled_actual,
        "line": pooled_line,
        "flat_decimal_odds": config.betting.flat_decimal_odds,
    }
    betting_sweep = betting_threshold_sweep(
        predicted_edge=pooled_edge,
        thresholds=config.betting.edge_thresholds,
        **betting_kwargs,
    )
    betting_primary = evaluate_betting(
        predicted_edge=pooled_edge,
        min_edge=config.betting.primary_edge_threshold,
        **betting_kwargs,
    )

    line_comparison = build_line_comparison(
        y_pred=predictions["y_pred"].to_numpy(dtype=float),
        target_line=pooled_line,
        actual_total=pooled_actual,
        lines=collect_comparison_lines(
            df_dev,
            config,
            target_line_col=target_line_col,
            positions=predictions["row_in_dev"].to_numpy(),
        ),
        config=config,
    )

    return CrossValidationBettingResult(
        n_folds=len(splits),
        n_games=int(len(predictions)),
        n_unique_games=int(predictions["row_in_dev"].nunique()),
        mae=float(
            mean_absolute_error(predictions["y_true"], predictions["y_pred"])
        ),
        rmse=float(
            np.sqrt(mean_squared_error(predictions["y_true"], predictions["y_pred"]))
        ),
        mean_fold_mae=float(fold_metrics["mae"].mean()),
        betting_primary=betting_primary,
        betting_sweep=betting_sweep,
        fold_metrics=fold_metrics,
        predictions=predictions,
        line_comparison=line_comparison,
        n_profitable_folds=int((fold_metrics["roi"].fillna(0.0) > 0).sum()),
        random_state=resolved_random_state,
    )
