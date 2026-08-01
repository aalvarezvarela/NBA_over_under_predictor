"""Strategy/adapter layer resolving the asymmetry between optuna_total_points.py
and optuna_error_line.py.

Confirmed by direct inspection: optuna_total_points.py always needs `line_col`
(it pulls the betting line from X[line_col] for OU-accuracy scoring) and has
no sample-weight support at all. optuna_error_line.py never takes `line_col`
(it scores directly off the sign of the predicted error) but fully supports
recency sample-weighting (sample_weight/sample_weight_dates/
sample_weight_lambda/tune_sample_weight_lambda).

get_strategy(config) is the only branch point in this whole package -- every
other module calls strategy.<method>(...) and never imports optuna_total_points
or optuna_error_line directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling import optuna_error_line as _error_line
from nba_ou.modeling import optuna_total_points as _total_points
from nba_ou.modeling.modeling import build_recency_sample_weights
from nba_ou.modeling.scorers import (
    evaluate_error_thresholds,
    evaluate_total_points_thresholds,
)
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

from training_pipeline.config import (
    ExperimentConfig,
    SampleWeightConfig,
    SearchSpaceConfig,
    TargetFamily,
)


class TargetFamilyStrategy(Protocol):
    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        config: ExperimentConfig,
        dates: pd.Series | None = None,
    ) -> optuna.Study: ...

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial: ...

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame: ...

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame: ...

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None,
        trial: optuna.trial.FrozenTrial | None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBRegressor: ...

    def evaluate_holdout(
        self,
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]: ...


def build_xgb_params(
    trial: optuna.Trial,
    space: SearchSpaceConfig,
    *,
    objective_name: str,
    random_state: int = 16,
) -> dict[str, Any]:
    """Sample XGBoost parameters from a configurable search space.

    The suggest_* calls are issued in exactly the same order, with the same
    names and distributions, as
    ``nba_ou.modeling.optuna_total_points.build_xgb_params_total_points``. That
    ordering matters: a seeded TPE sampler draws per-parameter in call order,
    so any reordering would change results even with identical ranges. A test
    asserts both builders yield identical draws under the same seed.
    """
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": objective_name,
        "eval_metric": "mae",
        "max_depth": trial.suggest_int(
            "max_depth", space.max_depth.low, space.max_depth.high, log=space.max_depth.log
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            space.min_child_weight.low,
            space.min_child_weight.high,
            log=space.min_child_weight.log,
        ),
        "gamma": trial.suggest_float(
            "gamma", space.gamma.low, space.gamma.high, log=space.gamma.log
        ),
        "subsample": trial.suggest_float(
            "subsample", space.subsample.low, space.subsample.high, log=space.subsample.log
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            space.colsample_bytree.low,
            space.colsample_bytree.high,
            log=space.colsample_bytree.log,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space.learning_rate.low,
            space.learning_rate.high,
            log=space.learning_rate.log,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", space.reg_alpha.low, space.reg_alpha.high, log=space.reg_alpha.log
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            space.reg_lambda.low,
            space.reg_lambda.high,
            log=space.reg_lambda.log,
        ),
        "n_estimators": space.n_estimators,
        "early_stopping_rounds": space.early_stopping_rounds,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


#: Optuna parameter name for the "weight at all?" decision.
USE_SAMPLE_WEIGHT_PARAM = "use_sample_weight"

#: Params belonging to the training protocol, not to XGBoost itself.
NON_XGB_TRIAL_PARAMS = frozenset({"sample_weight_lambda", USE_SAMPLE_WEIGHT_PARAM})


def resolve_final_params(
    trial: optuna.trial.FrozenTrial, config: ExperimentConfig
) -> tuple[dict[str, Any], int, float | None]:
    """Split a tuned trial into (xgb params, boosting rounds, decay rate)."""
    from nba_ou.modeling.optuna_total_points import get_trial_n_estimators

    params = {k: v for k, v in trial.params.items() if k not in NON_XGB_TRIAL_PARAMS}
    lambda_ = trial.params.get("sample_weight_lambda")

    # A trial that explicitly chose not to weight must not have weighting
    # reinstated by the config fallback below.
    chose_unweighted = trial.params.get(USE_SAMPLE_WEIGHT_PARAM) is False
    if chose_unweighted:
        lambda_ = None
    elif lambda_ is None and config.sample_weight.enabled:
        lambda_ = config.sample_weight.lambda_

    return params, get_trial_n_estimators(trial), lambda_


def fit_final_model(
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    params: dict[str, Any],
    n_estimators: int,
    config: ExperimentConfig,
    dates_dev: pd.Series | None = None,
    sample_weight_lambda: float | None = None,
) -> XGBRegressor:
    """Fit one model on fixed hyperparameters.

    Shared by both target families, which is what makes recency weighting
    available to TOTAL_POINTS: upstream's fit_best_xgb_total_points has no
    sample_weight parameter, so routing through it would silently drop the
    weights.
    """
    final_params = {
        **_build_static_params(config),
        **params,
        "n_estimators": n_estimators,
    }
    # No eval_set here, so early stopping cannot apply.
    final_params.pop("early_stopping_rounds", None)

    sample_weight = None
    if sample_weight_lambda is not None and dates_dev is not None:
        sample_weight = build_recency_sample_weights(
            dates_dev, lambda_=float(sample_weight_lambda)
        ).to_numpy(dtype=float)

    model = XGBRegressor(**final_params)
    model.fit(X_dev, y_dev, sample_weight=sample_weight, verbose=False)
    return model


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


def _resolve_trial_sample_weight_lambda(
    trial: optuna.Trial, sample_weight: SampleWeightConfig
) -> float | None:
    """Decay rate for this trial, or None for no weighting.

    When ``allow_unweighted`` is on, a categorical decides first whether to
    weight at all; the decay rate is only sampled if it says yes. That is a
    conditional search space, which TPE handles natively, and it lets "no
    weighting" compete as a real option instead of being unreachable at the
    bottom of a log-uniform range.
    """
    if not sample_weight.enabled:
        return None
    if not sample_weight.tune_lambda:
        return sample_weight.lambda_

    if sample_weight.allow_unweighted:
        use_weighting = trial.suggest_categorical(USE_SAMPLE_WEIGHT_PARAM, [True, False])
        if not use_weighting:
            return None

    low, high = sample_weight.lambda_bounds
    return float(trial.suggest_float("sample_weight_lambda", low, high, log=True))


def run_objective(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
    evaluate_fold: Callable[..., Any],
    dates: pd.Series | None = None,
) -> float:
    """Mean validation MAE across time-aware folds, minimised by Optuna.

    Mirrors the aggregation and ``user_attrs`` contract of the upstream
    objectives (``mean_mae``/``mean_rmse``/``mean_ou_acc``/
    ``median_best_iteration``/...), which downstream helpers such as
    ``select_best_trial_lexicographic`` and ``summarize_optuna_trials`` read.
    Per-fold metrics come from upstream's own ``evaluate_fold_*`` so metric
    definitions cannot drift.
    """
    params = build_xgb_params(
        trial, config.optuna.search_space, objective_name=config.optuna.objective_name
    )
    weight_lambda = _resolve_trial_sample_weight_lambda(trial, config.sample_weight)

    fold_metrics: list[Any] = []
    for fold_num, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        sample_weight = None
        if weight_lambda is not None and dates is not None:
            sample_weight = build_recency_sample_weights(
                dates.iloc[train_idx], lambda_=float(weight_lambda)
            ).to_numpy(dtype=float)

        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        fold_metrics.append(
            evaluate_fold(model, X_valid, y_valid, fold=fold_num, n_train=len(X_train))
        )

        trial.report(float(np.mean([m.mae for m in fold_metrics])), step=fold_num)
        if trial.should_prune():
            raise optuna.TrialPruned()

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(m, attr) for m in fold_metrics]))

    mean_mae = _mean("mae")
    if weight_lambda is not None:
        trial.set_user_attr("sample_weight_lambda", float(weight_lambda))
    trial.set_user_attr("mean_mae", mean_mae)
    trial.set_user_attr("mean_rmse", _mean("rmse"))
    trial.set_user_attr("mean_r2", _mean("r2"))
    trial.set_user_attr("mean_ou_acc", _mean("ou_accuracy"))
    trial.set_user_attr("mean_ou_acc_edge_2", _mean("ou_accuracy_edge_2"))
    trial.set_user_attr("mean_ou_acc_edge_3", _mean("ou_accuracy_edge_3"))
    trial.set_user_attr("mean_ou_acc_edge_4", _mean("ou_accuracy_edge_4"))
    trial.set_user_attr(
        "mean_best_iteration",
        int(round(np.mean([m.best_iteration for m in fold_metrics]))),
    )
    trial.set_user_attr(
        "median_best_iteration", int(np.median([m.best_iteration for m in fold_metrics]))
    )
    trial.set_user_attr("fold_metrics", [vars(m) for m in fold_metrics])
    return mean_mae


def _create_study(config: ExperimentConfig) -> optuna.Study:
    """One study construction path, persistent or in-memory."""
    kwargs: dict[str, Any] = {}
    if config.optuna.persistent_storage:
        kwargs = {
            "storage": _persistent_storage_url(config),
            "load_if_exists": config.optuna.load_if_exists,
        }
    return optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=16),
        pruner=MedianPruner(n_warmup_steps=5),
        study_name=config.resolved_study_name,
        **kwargs,
    )


def _optimize(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    config: ExperimentConfig,
) -> optuna.Study:
    study.optimize(
        objective,
        n_trials=config.optuna.n_trials,
        timeout=config.optuna.timeout,
        n_jobs=1,
        show_progress_bar=True,
    )
    return study


def _persistent_storage_url(config: ExperimentConfig) -> str:
    """A stable (non-timestamped) location so re-running the same config
    resumes the same study (that's the point of `load_if_exists=True`) rather
    than starting a fresh study every run the way the timestamped per-run
    artifact directory does.

    The filename includes ExperimentConfig.fingerprint(), so a study can only
    be resumed by a config whose trials are actually comparable: change the
    CSV, cleaning thresholds, fold layout or XGBoost objective and you get a
    new database instead of silently appending incomparable trials to the old
    one. Renaming the experiment or changing n_trials/timeout still resumes.
    """
    db_dir = config.experiment_root_dir / "_optuna_studies"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_name = (
        f"{config.experiment_name}_{config.target_family.value}"
        f"_{config.fingerprint()}.db"
    )
    db_path = db_dir / db_name
    return f"sqlite:///{db_path.resolve().as_posix()}"


@dataclass
class TotalPointsStrategy:
    line_col: str

    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        config: ExperimentConfig,
        dates: pd.Series | None = None,
    ) -> optuna.Study:
        def evaluate_fold(
            model: XGBRegressor,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
            *,
            fold: int,
            n_train: int,
        ) -> Any:
            return _total_points.evaluate_fold_total_points(
                model, X_valid, y_valid, self.line_col, fold=fold, n_train=n_train
            )

        return _optimize(
            _create_study(config),
            lambda trial: run_objective(
                trial,
                X=X,
                y=y,
                splits=splits,
                config=config,
                evaluate_fold=evaluate_fold,
            ),
            config,
        )

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial:
        return _total_points.select_best_trial_lexicographic(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame:
        return _total_points.summarize_optuna_trials(study)

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame:
        return _total_points.summarize_lexicographic_candidates(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None = None,
        trial: optuna.trial.FrozenTrial | None = None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBRegressor:
        selected = trial if trial is not None else study.best_trial  # type: ignore[union-attr]
        params, n_estimators, lambda_ = resolve_final_params(selected, config)
        return fit_final_model(
            X_dev=X_dev,
            y_dev=y_dev,
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev,
            sample_weight_lambda=lambda_,
        )

    def evaluate_holdout(
        self,
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        return evaluate_total_points_thresholds(
            model=model,
            X_test=X_test,
            y_test_total=y_test,
            line_col=self.line_col,
        )


@dataclass
class LineErrorStrategy:
    sample_weight: SampleWeightConfig

    def _tune_sample_weight_kwargs(self, *, dates: pd.Series | None) -> dict[str, Any]:
        if not self.sample_weight.enabled:
            return {}
        kwargs: dict[str, Any] = {
            "sample_weight_lambda": self.sample_weight.lambda_,
            "tune_sample_weight_lambda": self.sample_weight.tune_lambda,
            "sample_weight_lambda_bounds": self.sample_weight.lambda_bounds,
        }
        if dates is not None:
            kwargs["sample_weight_dates"] = dates
        return kwargs

    def _fit_sample_weight_kwargs(self, *, dates: pd.Series | None) -> dict[str, Any]:
        if not self.sample_weight.enabled:
            return {}
        kwargs: dict[str, Any] = {"sample_weight_lambda": self.sample_weight.lambda_}
        if dates is not None:
            kwargs["sample_weight_dates"] = dates
        return kwargs

    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        config: ExperimentConfig,
        dates: pd.Series | None = None,
    ) -> optuna.Study:
        def evaluate_fold(
            model: XGBRegressor,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
            *,
            fold: int,
            n_train: int,
        ) -> Any:
            return _error_line.evaluate_fold_error_line(
                model, X_valid, y_valid, fold=fold, n_train=n_train
            )

        return _optimize(
            _create_study(config),
            lambda trial: run_objective(
                trial,
                X=X,
                y=y,
                splits=splits,
                config=config,
                evaluate_fold=evaluate_fold,
                dates=dates,
            ),
            config,
        )

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial:
        return _error_line.select_best_trial_lexicographic(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame:
        return _error_line.summarize_optuna_trials(study)

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame:
        return _error_line.summarize_lexicographic_candidates(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None = None,
        trial: optuna.trial.FrozenTrial | None = None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBRegressor:
        selected = trial if trial is not None else study.best_trial  # type: ignore[union-attr]
        params, n_estimators, lambda_ = resolve_final_params(selected, config)
        return fit_final_model(
            X_dev=X_dev,
            y_dev=y_dev,
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev,
            sample_weight_lambda=lambda_,
        )

    def evaluate_holdout(
        self,
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        return evaluate_error_thresholds(model=model, X_test=X_test, y_test_error=y_test)


def get_strategy(config: ExperimentConfig) -> TargetFamilyStrategy:
    if config.target_family == TargetFamily.TOTAL_POINTS:
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        return TotalPointsStrategy(line_col=config.line_col)
    return LineErrorStrategy(sample_weight=config.sample_weight)
