"""run_experiment(): the single high-level training method.

Composes every lower-level step in order. Every intermediate object remains
independently callable/inspectable -- a notebook can call run_experiment(config)
and still display() any intermediate table afterward (result.fold_info,
result.trials_df, result.holdout_result.threshold_results, ...), or skip
run_experiment entirely and call the lower-level functions one at a time, the
same way the repo's example notebooks already do.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.modeling import save_model_bundle
from xgboost import XGBRegressor

from training_pipeline import naming, tracking
from training_pipeline.backtest import (
    DailyBacktestResult,
    run_walk_forward_evaluation,
    xgb_params_from_trial,
)
from training_pipeline.baseline import (
    BaselineMetrics,
    compute_baseline_metrics_across_folds,
    compute_line_error_bias,
)
from training_pipeline.config import (
    ExperimentConfig,
    HoldoutEvaluation,
    RefitStrategy,
    TargetFamily,
)
from training_pipeline.data import (
    PreparedDataset,
    build_feature_matrix,
    prepare_dataset,
)
from training_pipeline.evaluation import (
    HoldoutEvaluationResult,
    evaluate_on_holdout,
)
from training_pipeline.splits import build_holdout_split, build_walk_forward_splits
from training_pipeline.tuning import fit_final_model, get_strategy


def _feature_target_subset(
    df_subset: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild X/y directly from a dev/test subset rather than slicing
    PreparedDataset.X/.y by index -- split_latest_dates_holdout resets the
    index on both halves it returns, so index-based alignment back to the
    full prepared frame would silently misalign rows.
    """
    target_col = "LINE_ERROR" if config.target_family == TargetFamily.LINE_ERROR else "TOTAL_POINTS"
    return build_feature_matrix(df_subset, target_col=target_col, exclude_cols=config.exclude_cols)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    prepared: PreparedDataset
    df_dev: pd.DataFrame
    df_test: pd.DataFrame
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_info: pd.DataFrame
    #: None when optuna.fixed_params skipped tuning.
    study: optuna.Study | None
    trials_df: pd.DataFrame | None
    candidates_df: pd.DataFrame | None
    selected_trial: optuna.trial.FrozenTrial | None
    #: The shipped model. None unless refit.train_production_model is on.
    model: XGBRegressor | None
    #: Set when holdout_evaluation == single_shot.
    holdout_result: HoldoutEvaluationResult | None
    #: Set when holdout_evaluation == daily_walk_forward (the default): the
    #: test period scored one day at a time, retraining each day.
    walk_forward_result: DailyBacktestResult | None
    baseline_cv: BaselineMetrics
    baseline_fold_df: pd.DataFrame
    run_dir: Path | None
    model_path: Path | None
    meta_path: Path | None


def run_experiment(
    config: ExperimentConfig, *, save_model: bool | None = None
) -> ExperimentResult:
    """Run one experiment end to end.

    ``save_model`` overrides ``refit.train_production_model`` when given;
    leave it None to let the config decide.
    """
    train_production = (
        config.refit.train_production_model if save_model is None else save_model
    )
    prepared = prepare_dataset(config)
    df_dev, df_test = build_holdout_split(prepared.df_full, config)

    X_dev, y_dev = _feature_target_subset(df_dev, config)
    X_test, y_test = _feature_target_subset(df_test, config)
    dates_dev = df_dev[config.data.date_col]

    # Resolve the output bundle path and check writability *before* tuning:
    # the name depends only on the training window's end date, which is known
    # now, so a refusal costs seconds instead of discarding a full Optuna run.
    train_date_max = df_dev[config.data.date_col].max()
    model_name: str | None = None
    model_out_dir: Path | None = None
    if train_production:
        model_name = naming.build_model_name(
            config, as_of=pd.Timestamp(train_date_max).date()
        )
        model_out_dir = naming.resolve_model_output_dir(config)
        naming.assert_model_bundle_is_writable(
            model_out_dir,
            model_name=model_name,
            overwrite_existing_model=config.overwrite_existing_model,
        )

    splits, fold_info = build_walk_forward_splits(df_dev, config)

    strategy = get_strategy(config)

    study: optuna.Study | None = None
    trials_df: pd.DataFrame | None = None
    candidates_df: pd.DataFrame | None = None
    if not config.optuna.skip_tuning:
        study = strategy.tune(
            X=X_dev, y=y_dev, splits=splits, config=config, dates=dates_dev
        )
        trials_df = strategy.summarize_trials(study)
        candidates_df = strategy.summarize_candidates(
            study,
            mae_tolerance_abs=config.optuna.mae_tolerance_abs,
            mae_tolerance_pct=config.optuna.mae_tolerance_pct,
        )

    baseline_fold_df, baseline_cv = compute_baseline_metrics_across_folds(
        df_dev, splits, baseline_line_col=prepared.baseline_line_col
    )

    # Fitted on dev rows only, then applied to the test period, so the
    # bias-corrected null never sees it.
    dev_line_error_bias = compute_line_error_bias(
        df_dev, baseline_line_col=prepared.baseline_line_col
    )

    # The hyperparameters to hold fixed for evaluation: either recovered from
    # a config that pinned them, or the ones Optuna just chose.
    selected_trial: optuna.trial.FrozenTrial | None = None
    reporting_trial: optuna.trial.FrozenTrial | None = None
    if config.optuna.skip_tuning:
        tuned_params = dict(config.optuna.fixed_params or {})
        tuned_n_estimators = int(config.optuna.fixed_n_estimators or 0)
        tuned_lambda = config.optuna.fixed_sample_weight_lambda
        if tuned_lambda is None and config.sample_weight.enabled:
            tuned_lambda = config.sample_weight.lambda_
    else:
        assert study is not None
        if config.refit.use_lexicographic_selection:
            selected_trial = strategy.select_best_trial(
                study,
                mae_tolerance_abs=config.optuna.mae_tolerance_abs,
                mae_tolerance_pct=config.optuna.mae_tolerance_pct,
            )
        reporting_trial = (
            selected_trial if selected_trial is not None else study.best_trial
        )
        tuned_params, tuned_n_estimators, tuned_lambda = xgb_params_from_trial(
            reporting_trial
        )

    holdout_result: HoldoutEvaluationResult | None = None
    walk_forward_result: DailyBacktestResult | None = None

    if config.holdout_evaluation == HoldoutEvaluation.DAILY_WALK_FORWARD:
        # Score the test period the way production runs: retrain once per game
        # day on everything available strictly before it (dev, plus test days
        # already played), predict only that day, then pool all days.
        walk_forward_result = run_walk_forward_evaluation(
            config,
            prepared=prepared,
            df_history=df_dev,
            df_evaluation=df_test,
            train_games=config.walk_forward.train_games,
            xgb_params=tuned_params,
            n_estimators=tuned_n_estimators,
            sample_weight_lambda=tuned_lambda,
            show_progress=config.backtest.show_progress,
        )
    else:
        # One model fitted on the dev window, predicting the whole test period
        # at once. Cheaper, but never absorbs completed test days.
        single_shot_model = fit_final_model(
            X_dev=X_dev,
            y_dev=y_dev,
            params=tuned_params,
            n_estimators=tuned_n_estimators,
            config=config,
            dates_dev=dates_dev,
            sample_weight_lambda=tuned_lambda,
        )
        holdout_result = evaluate_on_holdout(
            strategy,
            single_shot_model,
            X_test=X_test,
            y_test=y_test,
            df_test_full=df_test,
            baseline_line_col=prepared.baseline_line_col,
            target_line_col=prepared.target_line_col,
            dev_line_error_bias=dev_line_error_bias,
            config=config,
        )

    # --- production model -------------------------------------------------
    # Fitted on the FULL dataset (dev + test) because production holds nothing
    # back. Never scored here: the walk-forward above measures the procedure
    # that produces it.
    model: XGBRegressor | None = None
    if train_production:
        X_full, y_full = _feature_target_subset(prepared.df_full, config)
        dates_full = prepared.df_full[config.data.date_col]
        train_games = config.walk_forward.train_games
        if config.refit.strategy == RefitStrategy.ROLLING_WINDOW and train_games:
            X_full = X_full.tail(train_games)
            y_full = y_full.loc[X_full.index]
            dates_full = dates_full.loc[X_full.index]
        model = fit_final_model(
            X_dev=X_full,
            y_dev=y_full,
            params=tuned_params,
            n_estimators=tuned_n_estimators,
            config=config,
            dates_dev=dates_full,
            sample_weight_lambda=tuned_lambda,
        )

    # No CV metrics exist when tuning was skipped -- nothing was cross-validated.
    cv_mae: float | None = None
    cv_rmse = None
    cv_ou_acc = None
    if reporting_trial is not None:
        reported_mae = reporting_trial.user_attrs.get("mean_mae", reporting_trial.value)
        cv_mae = None if reported_mae is None else float(reported_mae)
        cv_rmse = reporting_trial.user_attrs.get("mean_rmse")
        cv_ou_acc = reporting_trial.user_attrs.get("mean_ou_acc")

    # One metric surface regardless of evaluation mode, so artifacts and the
    # leaderboard read the same either way.
    if holdout_result is not None:
        test_mae = holdout_result.mae
        test_rmse = holdout_result.rmse
        test_r2 = holdout_result.r2
        test_ou_acc = holdout_result.ou_accuracy
        test_baseline = holdout_result.baseline_holdout
        test_betting_primary = holdout_result.betting_primary
        test_betting_sweep = holdout_result.betting_sweep
        test_predictions = holdout_result.predictions_df
    else:
        assert walk_forward_result is not None
        test_mae = walk_forward_result.mae
        test_rmse = walk_forward_result.rmse
        test_r2 = walk_forward_result.r2
        # The walk-forward's directional accuracy at the primary edge; there is
        # no single-model threshold sweep to read a zero-edge accuracy from.
        test_ou_acc = walk_forward_result.betting_primary.win_rate or float("nan")
        test_baseline = walk_forward_result.baseline
        test_betting_primary = walk_forward_result.betting_primary
        test_betting_sweep = walk_forward_result.betting_sweep
        test_predictions = walk_forward_result.predictions

    run_dir: Path | None = None
    model_path: Path | None = None
    meta_path: Path | None = None

    if config.save_experiment_artifacts:
        created_at = datetime.now(tz=UTC)
        # Short, citable identity for this run. The config fingerprint already
        # answers "was this the same setup"; this answers "which run was it".
        experiment_id = uuid.uuid4().hex[:12]
        run_dir = tracking.create_experiment_run_dir(
            config.experiment_name,
            root_dir=config.experiment_root_dir,
            timestamp=created_at,
        )
        holdout_dates = pd.to_datetime(df_test[config.data.date_col])
        tracking.save_experiment_metadata(
            run_dir,
            {
                "experiment_id": experiment_id,
                "experiment_name": config.experiment_name,
                "training_version": config.training_version,
                # Research log: why this run exists, and what it should be read
                # alongside.
                "comparison_group": config.comparison_group,
                "hypothesis": config.hypothesis,
                "tags": list(config.tags),
                # Provenance of the bytes actually trained on.
                "data_version": config.data.data_version,
                "dataset_checksum": prepared.dataset_checksum,
                "csv_path": str(config.data.csv_path),
                "created_at": created_at.isoformat(),
                "target_family": config.target_family.value,
                # Resolved (not raw) labels, so the leaderboard can group runs
                # even when these were auto-derived rather than set explicitly.
                "window_dir_label": config.resolved_window_dir_label,
                "window_name_label": config.resolved_window_name_label,
                "config_fingerprint": config.fingerprint(),
                # The evaluation cohort. Two runs scored on different holdout
                # windows are not directly comparable, so the leaderboard
                # surfaces this rather than silently ranking them together.
                "holdout_start": pd.Timestamp(holdout_dates.min()).isoformat(),
                "holdout_end": pd.Timestamp(holdout_dates.max()).isoformat(),
                "holdout_n_games": int(len(df_test)),
                "holdout_evaluation": config.holdout_evaluation.value,
                "trained_production_model": bool(train_production),
            },
        )
        tracking.save_config_snapshot(run_dir, config)
        tracking.save_feature_schema(run_dir, prepared.feature_names)
        if study is not None:
            tracking.save_optuna_artifacts(
                run_dir=run_dir,
                study=study,
                trials_df=trials_df,
                selected_trial=selected_trial,
                candidates_df=candidates_df,
            )
        tracking.save_final_test_artifacts(
            run_dir=run_dir,
            metrics={
                "cv": {"mae": cv_mae, "rmse": cv_rmse, "ou_acc": cv_ou_acc},
                "holdout": {
                    "mae": test_mae,
                    "rmse": test_rmse,
                    "r2": test_r2,
                    "ou_acc": test_ou_acc,
                },
            },
            predictions_df=test_predictions,
            threshold_results_df=(
                holdout_result.threshold_results if holdout_result is not None else None
            ),
        )
        tracking.save_baseline_metrics(
            run_dir,
            baseline_cv=baseline_cv,
            baseline_fold_df=baseline_fold_df,
            baseline_holdout=test_baseline,
        )
        tracking.save_betting_metrics(
            run_dir,
            betting_sweep=test_betting_sweep,
            betting_primary=test_betting_primary,
            baseline_bias_corrected=(
                holdout_result.baseline_bias_corrected
                if holdout_result is not None
                else test_baseline
            ),
            baseline_bias_corrected_betting=(
                holdout_result.baseline_bias_corrected_betting
                if holdout_result is not None
                else test_betting_primary
            ),
            dev_line_error_bias=dev_line_error_bias,
        )
        if walk_forward_result is not None:
            tracking.save_backtest_artifacts(
                run_dir,
                summary=walk_forward_result.summary(),
                daily_results_df=walk_forward_result.daily_results,
                predictions_df=walk_forward_result.predictions,
                betting_sweep=walk_forward_result.betting_sweep,
            )

    if train_production:
        assert model_name is not None and model_out_dir is not None
        metadata = naming.build_model_bundle_metadata(
            config,
            model_name=model_name,
            best_params=tuned_params,
            selected_trial_number=(
                reporting_trial.number if reporting_trial is not None else None
            ),
            mean_best_iteration=tuned_n_estimators,
            median_best_iteration=tuned_n_estimators,
            train_games=config.walk_forward.train_games,
            cv_mae=cv_mae if cv_mae is not None else float("nan"),
            cv_rmse=cv_rmse,
            cv_ou_acc=cv_ou_acc,
            final_test_mae=test_mae,
            final_test_rmse=test_rmse,
            final_test_ou_acc=test_ou_acc,
            train_date_min=df_dev[config.data.date_col].min(),
            train_date_max=train_date_max,
        )
        model_path, meta_path = save_model_bundle(
            model=model,
            feature_names=prepared.feature_names,
            out_dir=model_out_dir,
            metadata=metadata,
        )

    return ExperimentResult(
        config=config,
        prepared=prepared,
        df_dev=df_dev,
        df_test=df_test,
        splits=splits,
        fold_info=fold_info,
        study=study,
        trials_df=trials_df,
        candidates_df=candidates_df,
        selected_trial=selected_trial,
        model=model,
        holdout_result=holdout_result,
        walk_forward_result=walk_forward_result,
        baseline_cv=baseline_cv,
        baseline_fold_df=baseline_fold_df,
        run_dir=run_dir,
        model_path=model_path,
        meta_path=meta_path,
    )
