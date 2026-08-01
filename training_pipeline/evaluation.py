"""Refit strategy + final holdout scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from training_pipeline.baseline import (
    BaselineMetrics,
    compute_baseline_metrics_for_rows,
    compute_bias_corrected_baseline_metrics,
)
from training_pipeline.betting import (
    BettingMetrics,
    betting_threshold_sweep,
    evaluate_betting,
)
from training_pipeline.config import ExperimentConfig, RefitStrategy, TargetFamily
from training_pipeline.tuning import TargetFamilyStrategy


def train_production_model(
    strategy: TargetFamilyStrategy,
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    dates_dev: pd.Series | None,
    study: optuna.Study,
    config: ExperimentConfig,
) -> tuple[XGBRegressor, optuna.trial.FrozenTrial | None]:
    """Fit the model that would actually be shipped.

    Called with the FULL dataset (dev + test), not just dev: nothing is held
    back in production, so the shipped model should see the freshest games.
    That also means this model is never itself scored -- the walk-forward
    evaluation measures the *procedure* that produces it.

    RefitStrategy.FULL_DATASET fits on everything using study.best_trial
    directly. ROLLING_WINDOW (default) optionally re-selects the trial
    lexicographically, then fits on the most recent
    ``walk_forward.train_games`` rows.
    """
    if config.refit.strategy == RefitStrategy.FULL_DATASET:
        model = strategy.fit_best(
            X_dev=X_dev,
            y_dev=y_dev,
            study=study,
            trial=None,
            config=config,
            dates_dev=dates_dev,
        )
        return model, None

    # Same window the CV folds used, so the final model is fitted on the
    # amount of history the hyperparameters were selected for.
    train_games = config.walk_forward.train_games

    selected_trial: optuna.trial.FrozenTrial | None = None
    if config.refit.use_lexicographic_selection:
        selected_trial = strategy.select_best_trial(
            study,
            mae_tolerance_abs=config.optuna.mae_tolerance_abs,
            mae_tolerance_pct=config.optuna.mae_tolerance_pct,
        )

    if train_games is not None:
        X_dev = X_dev.tail(train_games)
        y_dev = y_dev.loc[X_dev.index]
        dates_dev = dates_dev.loc[X_dev.index] if dates_dev is not None else None

    if selected_trial is not None:
        model = strategy.fit_best(
            X_dev=X_dev,
            y_dev=y_dev,
            study=None,
            trial=selected_trial,
            config=config,
            dates_dev=dates_dev,
        )
    else:
        model = strategy.fit_best(
            X_dev=X_dev,
            y_dev=y_dev,
            study=study,
            trial=None,
            config=config,
            dates_dev=dates_dev,
        )

    return model, selected_trial


@dataclass
class HoldoutEvaluationResult:
    mae: float
    rmse: float
    r2: float
    ou_accuracy: float
    baseline_holdout: BaselineMetrics
    threshold_results: pd.DataFrame
    predictions_df: pd.DataFrame
    #: Profit-oriented metrics across every configured edge threshold.
    betting_sweep: pd.DataFrame
    #: The row of ``betting_sweep`` at betting.primary_edge_threshold -- the
    #: number that answers "would this have made money".
    betting_primary: BettingMetrics
    #: "Trust the line + its historical drift" null, fitted on dev rows only.
    baseline_bias_corrected: BaselineMetrics
    baseline_bias_corrected_betting: BettingMetrics
    dev_line_error_bias: float


def _extract_zero_threshold_accuracy(threshold_results: pd.DataFrame) -> float:
    if "ou_betting_accuracy" in threshold_results.columns:
        accuracy_col, threshold_col = "ou_betting_accuracy", "threshold_abs_pred_edge_gt"
    else:
        accuracy_col, threshold_col = "directional_accuracy", "threshold_abs_pred_error_gt"

    zero_row = threshold_results.loc[threshold_results[threshold_col] == 0]
    if zero_row.empty:
        return float("nan")
    return float(zero_row.iloc[0][accuracy_col])


def _optional_price_column(df: pd.DataFrame, column: str | None) -> np.ndarray | None:
    """Decimal-odds column as an array, or None to fall back to a flat price.

    Missing entirely is fine (the column may not have survived cleaning);
    per-row gaps are handled downstream in betting._resolve_prices.
    """
    if not column or column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def evaluate_on_holdout(
    strategy: TargetFamilyStrategy,
    model: XGBRegressor,
    *,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_full: pd.DataFrame,
    baseline_line_col: str,
    target_line_col: str,
    dev_line_error_bias: float,
    config: ExperimentConfig,
) -> HoldoutEvaluationResult:
    y_true = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(X_test), dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    threshold_results, _ = strategy.evaluate_holdout(model, X_test, y_test, config)
    ou_accuracy = _extract_zero_threshold_accuracy(threshold_results)

    baseline_holdout = compute_baseline_metrics_for_rows(
        df_test_full,
        np.arange(len(df_test_full)),
        baseline_line_col=baseline_line_col,
    )

    baseline_line = pd.to_numeric(
        df_test_full[baseline_line_col], errors="coerce"
    ).to_numpy(dtype=float)

    # baseline_pred must live in the SAME space as y_true/y_pred so the three
    # columns are directly comparable in the saved parquet. For a TOTAL_POINTS
    # run that space is points (the line itself); for a LINE_ERROR run it is
    # error-vs-line, where "trust the line" means predicting exactly 0. The
    # raw line is kept separately as baseline_line.
    if config.target_family == TargetFamily.LINE_ERROR:
        baseline_pred = np.zeros_like(y_pred)
    else:
        baseline_pred = baseline_line

    # --- Betting evaluation -------------------------------------------------
    # Bets settle against the line the target is defined relative to, which is
    # not necessarily baseline_line_col (that one may point at an alternative
    # consensus line purely for the MAE comparison).
    actual_total = pd.to_numeric(
        df_test_full["TOTAL_POINTS"], errors="coerce"
    ).to_numpy(dtype=float)
    target_line = pd.to_numeric(
        df_test_full[target_line_col], errors="coerce"
    ).to_numpy(dtype=float)

    # Predicted points relative to the line: for LINE_ERROR the prediction
    # already is that quantity; for TOTAL_POINTS subtract the line.
    if config.target_family == TargetFamily.LINE_ERROR:
        predicted_edge = y_pred
    else:
        predicted_edge = y_pred - target_line

    over_prices = _optional_price_column(df_test_full, config.betting.over_price_col)
    under_prices = _optional_price_column(df_test_full, config.betting.under_price_col)

    betting_kwargs = {
        "actual_total": actual_total,
        "line": target_line,
        "flat_decimal_odds": config.betting.flat_decimal_odds,
        "decimal_odds_over": over_prices,
        "decimal_odds_under": under_prices,
    }

    betting_sweep = betting_threshold_sweep(
        predicted_edge=predicted_edge,
        thresholds=config.betting.edge_thresholds,
        **betting_kwargs,
    )
    betting_primary = evaluate_betting(
        predicted_edge=predicted_edge,
        min_edge=config.betting.primary_edge_threshold,
        **betting_kwargs,
    )

    baseline_bias_corrected = compute_bias_corrected_baseline_metrics(
        df_test_full, baseline_line_col=baseline_line_col, bias=dev_line_error_bias
    )
    # The bias-corrected null bets the same side on every game with a constant
    # edge, which makes it all-or-nothing under a min-edge filter: it would
    # place either every bet or none, depending purely on whether the drift
    # happens to exceed the threshold. It is a fixed (non-selective) strategy,
    # so it is scored on every candidate game and compared against the model's
    # selective ROI.
    baseline_bias_corrected_betting = evaluate_betting(
        predicted_edge=np.full_like(actual_total, dev_line_error_bias),
        min_edge=0.0,
        **betting_kwargs,
    )

    predictions_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "baseline_pred": baseline_pred,
            "baseline_line": baseline_line,
            "target_line": target_line,
            "predicted_edge": predicted_edge,
            "TOTAL_POINTS": actual_total,
            config.data.date_col: df_test_full[config.data.date_col].to_numpy(),
        }
    )

    return HoldoutEvaluationResult(
        mae=mae,
        rmse=rmse,
        r2=r2,
        ou_accuracy=ou_accuracy,
        baseline_holdout=baseline_holdout,
        threshold_results=threshold_results,
        predictions_df=predictions_df,
        betting_sweep=betting_sweep,
        betting_primary=betting_primary,
        baseline_bias_corrected=baseline_bias_corrected,
        baseline_bias_corrected_betting=baseline_bias_corrected_betting,
        dev_line_error_bias=dev_line_error_bias,
    )
