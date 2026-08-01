"""Daily walk-forward backtest: the closest thing to a production dry run.

For every game-day in the backtest window the model is refitted on all games
that finished strictly before that day -- including earlier backtest days,
which become training data once played -- and then predicts only that day's
games. No result is visible to the model before it would have been in real
life, so the resulting predictions are genuinely out-of-sample in a way that a
single train/test split is not.

The day-by-day loop itself is
``nba_ou.modeling.modeling.evaluate_day_by_day_walk_forward``; this module
supplies the fit/predict closure, sizes the window, and scores the pooled
predictions with the same betting layer used elsewhere.

Hyperparameters are fixed across all days. Re-tuning daily would be
unrealistic and prohibitively slow (one Optuna study per day). The cost of
this choice: if those hyperparameters were selected on data overlapping the
backtest window, the backtest inherits that selection bias. Tune on data
preceding the window to keep it honest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.modeling import (
    build_recency_sample_weights,
    evaluate_day_by_day_walk_forward,
    split_latest_dates_holdout,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from training_pipeline.baseline import BaselineMetrics, compute_baseline_metrics
from training_pipeline.betting import (
    BettingMetrics,
    betting_threshold_sweep,
    evaluate_betting,
)
from training_pipeline.config import ExperimentConfig, TargetFamily
from training_pipeline.data import (
    PreparedDataset,
    build_feature_matrix,
    prepare_dataset,
)
from training_pipeline.tuning import (  # noqa: E402
    NON_XGB_TRIAL_PARAMS as _NON_XGB_TRIAL_PARAMS,
)
from training_pipeline.tuning import USE_SAMPLE_WEIGHT_PARAM  # noqa: E402


def xgb_params_from_trial(
    trial: optuna.trial.FrozenTrial,
) -> tuple[dict[str, Any], int, float | None]:
    """Extract fixed hyperparameters from a tuned Optuna trial.

    Returns ``(xgb_params, n_estimators, sample_weight_lambda)``.
    ``sample_weight_lambda`` is separated out because it is a training-protocol
    parameter, not an XGBoost one -- passing it through to ``XGBRegressor``
    would be silently accepted and then ignored.
    """
    from nba_ou.modeling.optuna_total_points import get_trial_n_estimators

    params = {k: v for k, v in trial.params.items() if k not in _NON_XGB_TRIAL_PARAMS}
    lambda_ = trial.params.get("sample_weight_lambda")
    if trial.params.get(USE_SAMPLE_WEIGHT_PARAM) is False:
        lambda_ = None
    return params, get_trial_n_estimators(trial), lambda_


def _build_static_params(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": config.optuna.objective_name,
        "eval_metric": "mae",
        "random_state": 16,
        "n_jobs": -1,
        "verbosity": 0,
    }


def _make_fit_and_predict(
    config: ExperimentConfig,
    *,
    target_col: str,
    xgb_params: dict[str, Any],
    n_estimators: int,
    sample_weight_lambda: float | None,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Closure fitting one model per day and predicting that day's games."""
    final_params = {
        **_build_static_params(config),
        **xgb_params,
        "n_estimators": n_estimators,
    }
    # No eval_set exists inside the daily loop, so early stopping cannot apply.
    final_params.pop("early_stopping_rounds", None)
    date_col = config.data.date_col

    def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        X_train, y_train = build_feature_matrix(
            train_df, target_col=target_col, exclude_cols=config.exclude_cols
        )
        X_test, _ = build_feature_matrix(
            test_df, target_col=target_col, exclude_cols=config.exclude_cols
        )

        sample_weight = None
        if sample_weight_lambda is not None:
            sample_weight = build_recency_sample_weights(
                train_df[date_col], lambda_=float(sample_weight_lambda)
            ).to_numpy(dtype=float)

        model = XGBRegressor(**final_params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        return np.asarray(model.predict(X_test), dtype=float)

    return fit_and_predict


@dataclass
class DailyBacktestResult:
    """Pooled outcome of a daily walk-forward backtest."""

    n_days: int
    n_games: int
    backtest_start: pd.Timestamp
    backtest_end: pd.Timestamp
    mae: float
    rmse: float
    r2: float
    mean_daily_mae: float
    baseline: BaselineMetrics
    betting_primary: BettingMetrics
    betting_sweep: pd.DataFrame
    #: One row per game-day: metric, training-set size and its date range.
    daily_results: pd.DataFrame
    #: One row per game with y_true/y_pred/line/edge, in chronological order.
    predictions: pd.DataFrame
    xgb_params: dict[str, Any]
    n_estimators: int
    sample_weight_lambda: float | None

    def summary(self) -> dict[str, Any]:
        return {
            "n_days": self.n_days,
            "n_games": self.n_games,
            "backtest_start": self.backtest_start.isoformat(),
            "backtest_end": self.backtest_end.isoformat(),
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mean_daily_mae": self.mean_daily_mae,
            "baseline_mae": self.baseline.mae,
            "baseline_rmse": self.baseline.rmse,
            "mae_improvement_over_baseline_pct": (
                (self.baseline.mae - self.mae) / self.baseline.mae
                if self.baseline.mae
                else None
            ),
            "roi": self.betting_primary.roi,
            "n_bets": self.betting_primary.n_bets,
            "win_rate": self.betting_primary.win_rate,
            "break_even_rate": self.betting_primary.break_even_rate,
            "is_significant": self.betting_primary.is_significant,
            "profit_units": self.betting_primary.profit_units,
            "xgb_params": self.xgb_params,
            "n_estimators": self.n_estimators,
            "sample_weight_lambda": self.sample_weight_lambda,
        }


def run_daily_backtest(
    config: ExperimentConfig,
    *,
    prepared: PreparedDataset | None = None,
    xgb_params: dict[str, Any] | None = None,
    n_estimators: int | None = None,
    sample_weight_lambda: float | None = None,
) -> DailyBacktestResult:
    """Run the daily walk-forward backtest described in BacktestConfig.

    ``prepared`` may be passed to reuse an already-loaded dataset; otherwise it
    is built from ``config`` (which excludes playoffs by default). Explicit
    ``xgb_params``/``n_estimators`` override BacktestConfig; pass the output of
    :func:`xgb_params_from_trial` to backtest a tuned configuration.
    """
    if prepared is None:
        prepared = prepare_dataset(config)

    df_history, df_backtest = split_latest_dates_holdout(
        df=prepared.df_full,
        date_col=config.data.date_col,
        test_size=None,
        test_games=config.backtest.test_games,
    )
    return run_walk_forward_evaluation(
        config,
        prepared=prepared,
        df_history=df_history,
        df_evaluation=df_backtest,
        train_games=config.backtest.train_games,
        xgb_params=xgb_params or config.backtest.resolved_xgb_params(),
        n_estimators=n_estimators or config.backtest.resolved_n_estimators(),
        sample_weight_lambda=sample_weight_lambda,
        show_progress=config.backtest.show_progress,
    )


def run_walk_forward_evaluation(
    config: ExperimentConfig,
    *,
    prepared: PreparedDataset,
    df_history: pd.DataFrame,
    df_evaluation: pd.DataFrame,
    train_games: int | None,
    xgb_params: dict[str, Any],
    n_estimators: int,
    sample_weight_lambda: float | None = None,
    show_progress: bool = True,
) -> DailyBacktestResult:
    """Score ``df_evaluation`` one game-day at a time, retraining each day.

    The caller supplies the split, so this serves both the standalone backtest
    and ``run_experiment``'s evaluation of its own holdout period with
    Optuna-tuned hyperparameters.
    """
    target_col = (
        "LINE_ERROR" if config.target_family == TargetFamily.LINE_ERROR else "TOTAL_POINTS"
    )
    resolved_params = xgb_params
    resolved_n_estimators = n_estimators
    resolved_lambda = sample_weight_lambda
    if resolved_lambda is None and config.sample_weight.enabled:
        resolved_lambda = config.sample_weight.lambda_

    df_backtest = df_evaluation

    walk_forward = evaluate_day_by_day_walk_forward(
        df_history,
        df_backtest,
        fit_and_predict=_make_fit_and_predict(
            config,
            target_col=target_col,
            xgb_params=resolved_params,
            n_estimators=resolved_n_estimators,
            sample_weight_lambda=resolved_lambda,
        ),
        metric_fn=lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred)),
        target_col=target_col,
        date_col=config.data.date_col,
        max_games=train_games,
        metric_name="mae",
        show_progress=show_progress,
        progress_desc=f"Daily backtest ({config.experiment_name})",
    )

    # Re-attach the line and actual total for each predicted game.
    # ``row_in_test_final`` is a positional index into df_backtest.
    positions = walk_forward.predictions["row_in_test_final"].to_numpy()
    target_line = pd.to_numeric(
        df_backtest[prepared.target_line_col], errors="coerce"
    ).to_numpy(dtype=float)[positions]
    baseline_line = pd.to_numeric(
        df_backtest[prepared.baseline_line_col], errors="coerce"
    ).to_numpy(dtype=float)[positions]
    actual_total = pd.to_numeric(
        df_backtest["TOTAL_POINTS"], errors="coerce"
    ).to_numpy(dtype=float)[positions]

    y_true = walk_forward.predictions["y_true"].to_numpy(dtype=float)
    y_pred = walk_forward.predictions["y_pred"].to_numpy(dtype=float)

    predicted_edge = (
        y_pred if config.target_family == TargetFamily.LINE_ERROR else y_pred - target_line
    )

    betting_kwargs: dict[str, Any] = {
        "actual_total": actual_total,
        "line": target_line,
        "flat_decimal_odds": config.betting.flat_decimal_odds,
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

    baseline = compute_baseline_metrics(
        y_true_total_points=pd.Series(actual_total),
        baseline_line=pd.Series(baseline_line),
        line_col=prepared.baseline_line_col,
    )

    predictions = walk_forward.predictions.assign(
        target_line=target_line,
        baseline_line=baseline_line,
        TOTAL_POINTS=actual_total,
        predicted_edge=predicted_edge,
    )

    return DailyBacktestResult(
        n_days=int(len(walk_forward.daily_results)),
        n_games=int(len(predictions)),
        backtest_start=pd.Timestamp(walk_forward.daily_results["date"].min()),
        backtest_end=pd.Timestamp(walk_forward.daily_results["date"].max()),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        mean_daily_mae=float(walk_forward.mean_metric),
        baseline=baseline,
        betting_primary=betting_primary,
        betting_sweep=betting_sweep,
        daily_results=walk_forward.daily_results,
        predictions=predictions,
        xgb_params=resolved_params,
        n_estimators=resolved_n_estimators,
        sample_weight_lambda=resolved_lambda,
    )
