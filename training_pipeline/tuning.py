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
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from training_pipeline.calibration import brier_score, log_loss
from training_pipeline.config import (
    ExperimentConfig,
    PredictionStrategy,
    SampleWeightConfig,
    SearchSpaceConfig,
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
    random_state: int | None = None,
) -> XGBModel:
    """Fit one model on fixed hyperparameters.

    Shared by both target families, which is what makes recency weighting
    available to TOTAL_POINTS: upstream's fit_best_xgb_total_points has no
    sample_weight parameter, so routing through it would silently drop the
    weights.

    ``random_state`` overrides ``config.random_state`` for this fit alone --
    the mechanism behind repeating an evaluation under several seeds. It is
    applied after ``params`` for two reasons: ``_build_static_params`` has
    already seeded from the config (so without this every "different" seed
    would fit identically), and a user-supplied ``optuna.fixed_params`` may
    legitimately contain its own ``random_state``.
    """
    final_params = {
        **_build_static_params(config),
        **params,
        "n_estimators": n_estimators,
    }
    if random_state is not None:
        final_params["random_state"] = random_state
    # A trial's params were sampled under a regression eval_metric default in
    # build_xgb_params; the static params above already hold the right one for
    # this strategy, so re-assert it after the merge.
    final_params["eval_metric"] = "logloss" if config.is_classifier else "mae"
    # No eval_set here, so early stopping cannot apply.
    final_params.pop("early_stopping_rounds", None)

    sample_weight = None
    if sample_weight_lambda is not None and dates_dev is not None:
        sample_weight = build_recency_sample_weights(
            dates_dev, lambda_=float(sample_weight_lambda)
        ).to_numpy(dtype=float)

    estimator_cls = XGBClassifier if config.is_classifier else XGBRegressor
    model = estimator_cls(**final_params)
    model.fit(X_dev, y_dev, sample_weight=sample_weight, verbose=False)
    return model


def _build_static_params(
    config: ExperimentConfig, *, random_state: int | None = None
) -> dict[str, Any]:
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": config.optuna.objective_name,
        # Early stopping and Optuna both track the loss that matches the model
        # class. Leaving "mae" on a classifier would early-stop on a metric
        # that ignores probability quality entirely.
        "eval_metric": "logloss" if config.is_classifier else "mae",
        "random_state": config.random_state if random_state is None else random_state,
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
        trial,
        config.optuna.search_space,
        objective_name=config.optuna.objective_name,
        random_state=config.random_state,
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
        sampler=TPESampler(seed=config.random_state),
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
        f"{config.experiment_name}_{config.family.value}"
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
                # Required, not optional: run_objective skips weighting entirely
                # when dates are absent, so omitting this would let Optuna draw
                # a sample_weight_lambda, record it on the trial, and score the
                # trial on unweighted fits -- selecting a training option it
                # never tested, which the final model would then apply.
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


@dataclass
class ClassifierFoldMetrics:
    """Per-fold classifier metrics, the analogue of upstream's fold objects."""

    fold: int
    n_train: int
    n_valid: int
    log_loss: float
    brier: float
    #: Fraction of validation games whose side the model called correctly. Every
    #: row has a definite answer here -- pushes were removed when the label was
    #: built -- so this needs no line and no push handling.
    ou_accuracy: float
    #: Outcome of actually betting the fold at the primary EV threshold.
    n_bets: int
    roi: float
    best_iteration: int


def _classifier_fold_metrics(
    model: XGBClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    fold: int,
    n_train: int,
    flat_decimal_odds: float,
    ev_threshold: float,
) -> ClassifierFoldMetrics:
    y = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=float)
    p_over = np.asarray(model.predict_proba(X_valid), dtype=float)[:, 1]

    ev_over = p_over * flat_decimal_odds - 1.0
    ev_under = (1.0 - p_over) * flat_decimal_odds - 1.0
    bet_over = ev_over >= ev_under
    placed = np.maximum(ev_over, ev_under) > ev_threshold

    # Pushes were dropped when the label was built, so a correct side is simply
    # a correct label call.
    correct = np.where(bet_over, y == 1.0, y == 0.0)

    n_bets = int(placed.sum())
    if n_bets:
        wins = int(correct[placed].sum())
        profit = wins * (flat_decimal_odds - 1.0) - (n_bets - wins)
        roi = profit / n_bets
    else:
        roi = 0.0

    best_iteration = int(getattr(model, "best_iteration", 0) or 0) + 1

    return ClassifierFoldMetrics(
        fold=fold,
        n_train=n_train,
        n_valid=int(len(y)),
        log_loss=log_loss(y, p_over),
        brier=brier_score(y, p_over),
        ou_accuracy=float(correct.mean()) if len(correct) else float("nan"),
        n_bets=n_bets,
        roi=roi,
        best_iteration=best_iteration,
    )


def run_classifier_objective(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
    dates: pd.Series | None = None,
) -> float:
    """Mean validation LOG LOSS across time-aware folds, minimised by Optuna.

    Log loss rather than accuracy because it scores the probability, not just
    the side: calling a loser at 51% is barely penalised, calling it at 95% is
    punished hard, and accuracy cannot tell those apart. The betting rule
    compares the probability against a break-even rate, so probability quality
    is exactly what needs optimising.

    Keep the scale in mind when reading trial values: on a ~50/50 outcome a
    perfectly calibrated 55% model scores 0.68814 against 0.69315 for a coin
    flip. Differences between good and bad trials live in the third decimal.
    """
    params = build_xgb_params(
        trial,
        config.optuna.search_space,
        objective_name=config.optuna.objective_name,
        random_state=config.random_state,
    )
    params["eval_metric"] = "logloss"
    weight_lambda = _resolve_trial_sample_weight_lambda(trial, config.sample_weight)

    fold_metrics: list[ClassifierFoldMetrics] = []
    for fold_num, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        sample_weight = None
        if weight_lambda is not None and dates is not None:
            sample_weight = build_recency_sample_weights(
                dates.iloc[train_idx], lambda_=float(weight_lambda)
            ).to_numpy(dtype=float)

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        fold_metrics.append(
            _classifier_fold_metrics(
                model,
                X_valid,
                y_valid,
                fold=fold_num,
                n_train=len(X_train),
                flat_decimal_odds=config.betting.flat_decimal_odds,
                ev_threshold=config.betting.primary_ev_threshold,
            )
        )

        trial.report(float(np.mean([m.log_loss for m in fold_metrics])), step=fold_num)
        if trial.should_prune():
            raise optuna.TrialPruned()

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(m, attr) for m in fold_metrics]))

    mean_logloss = _mean("log_loss")
    if weight_lambda is not None:
        trial.set_user_attr("sample_weight_lambda", float(weight_lambda))
    trial.set_user_attr("mean_logloss", mean_logloss)
    trial.set_user_attr("mean_brier", _mean("brier"))
    trial.set_user_attr("mean_ou_acc", _mean("ou_accuracy"))
    trial.set_user_attr("mean_roi", _mean("roi"))
    trial.set_user_attr("mean_n_bets", _mean("n_bets"))
    trial.set_user_attr(
        "mean_best_iteration",
        int(round(np.mean([m.best_iteration for m in fold_metrics]))),
    )
    trial.set_user_attr(
        "median_best_iteration",
        int(np.median([m.best_iteration for m in fold_metrics])),
    )
    trial.set_user_attr("fold_metrics", [vars(m) for m in fold_metrics])
    return mean_logloss


def _completed_classifier_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not trials:
        raise ValueError("No completed trials to select from.")
    return trials


def select_best_classifier_trial_lexicographic(
    study: optuna.Study,
    *,
    logloss_tolerance_abs: float | None = 0.002,
) -> optuna.trial.FrozenTrial:
    """Best log loss within tolerance, then the best betting outcome.

    Mirrors the regressors' lexicographic selection, and matters more here.
    Simulated at 600 validation games, log loss ranks a truly-53% trial above a
    truly-52% one only 64% of the time -- barely better than chance. Trusting
    that ordering outright would be selecting on noise, so the tolerance is
    deliberately wide relative to the metric's range and the real choice is
    made by the secondary criterion.
    """
    trials = _completed_classifier_trials(study)
    best = min(float(t.value) for t in trials)  # type: ignore[arg-type]
    cutoff = best + (logloss_tolerance_abs or 0.0)

    candidates = [
        t
        for t in trials
        if float(t.user_attrs.get("mean_logloss", t.value)) <= cutoff  # type: ignore[arg-type]
    ]
    if not candidates:
        candidates = trials

    def _key(trial: optuna.trial.FrozenTrial) -> tuple[float, float, float]:
        return (
            -float(trial.user_attrs.get("mean_ou_acc", 0.0)),
            -float(trial.user_attrs.get("mean_roi", 0.0)),
            float(trial.user_attrs.get("mean_logloss", trial.value)),  # type: ignore[arg-type]
        )

    return min(candidates, key=_key)


def summarize_classifier_trials(study: optuna.Study) -> pd.DataFrame:
    rows = [
        {
            "trial": trial.number,
            "state": trial.state.name,
            "value_logloss": trial.value,
            **{
                key: trial.user_attrs.get(key)
                for key in (
                    "mean_logloss",
                    "mean_brier",
                    "mean_ou_acc",
                    "mean_roi",
                    "mean_n_bets",
                    "median_best_iteration",
                    "sample_weight_lambda",
                )
            },
            **trial.params,
        }
        for trial in study.trials
    ]
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def summarize_classifier_candidates(
    study: optuna.Study, *, logloss_tolerance_abs: float | None = 0.002
) -> pd.DataFrame:
    """The trials that survived the log-loss tolerance, ranked as selected."""
    trials = _completed_classifier_trials(study)
    best = min(float(t.value) for t in trials)  # type: ignore[arg-type]
    cutoff = best + (logloss_tolerance_abs or 0.0)

    rows = [
        {
            "trial": trial.number,
            "mean_logloss": trial.user_attrs.get("mean_logloss", trial.value),
            "mean_brier": trial.user_attrs.get("mean_brier"),
            "mean_ou_acc": trial.user_attrs.get("mean_ou_acc"),
            "mean_roi": trial.user_attrs.get("mean_roi"),
            "mean_n_bets": trial.user_attrs.get("mean_n_bets"),
            "within_tolerance": (
                float(trial.user_attrs.get("mean_logloss", trial.value)) <= cutoff  # type: ignore[arg-type]
            ),
        }
        for trial in trials
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["within_tolerance", "mean_ou_acc"], ascending=[False, False]
    ).reset_index(drop=True)


@dataclass
class OverUnderClassifierStrategy:
    """Predicts P(OVER) directly instead of a total or an error.

    Deliberately does NOT reuse upstream's optuna_total_points/optuna_error_line
    helpers: those are regression-only throughout, from the objective down to
    ``select_best_trial_lexicographic``, which reads ``mean_mae``.
    """

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
        return _optimize(
            _create_study(config),
            lambda trial: run_classifier_objective(
                trial,
                X=X,
                y=y,
                splits=splits,
                config=config,
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
        # The MAE tolerances are ignored: this strategy is selected on log loss,
        # whose scale is unrelated. The parameters stay in the signature so the
        # strategy remains interchangeable with the regressors for callers.
        del mae_tolerance_abs, mae_tolerance_pct
        return select_best_classifier_trial_lexicographic(
            study, logloss_tolerance_abs=self._logloss_tolerance
        )

    #: Set by get_strategy from the config so select_best_trial, which the
    #: Protocol pins to MAE-shaped arguments, can still see the right tolerance.
    _logloss_tolerance: float | None = 0.002

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame:
        return summarize_classifier_trials(study)

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame:
        del mae_tolerance_abs, mae_tolerance_pct
        return summarize_classifier_candidates(
            study, logloss_tolerance_abs=self._logloss_tolerance
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
    ) -> XGBModel:
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
        model: XGBModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """EV-threshold sweep over the holdout.

        Returns the same (table, predictions) shape as the regressors so the
        holdout path stays uniform, but the threshold column is in EV units and
        the returned array is P(OVER), not a points prediction.
        """
        y = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
        p_over = np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
        odds = config.betting.flat_decimal_odds

        ev_over = p_over * odds - 1.0
        ev_under = (1.0 - p_over) * odds - 1.0
        bet_over = ev_over >= ev_under
        score = np.maximum(ev_over, ev_under)
        correct = np.where(bet_over, y == 1.0, y == 0.0)

        rows = []
        for threshold in config.betting.ev_thresholds:
            placed = score > threshold
            n = int(placed.sum())
            rows.append(
                {
                    "threshold_min_ev": threshold,
                    "n_bets": n,
                    "ou_betting_accuracy": (
                        float(correct[placed].mean()) if n else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows), p_over


def get_strategy(config: ExperimentConfig) -> TargetFamilyStrategy:
    if config.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        return OverUnderClassifierStrategy(
            line_col=config.line_col,
            _logloss_tolerance=config.optuna.logloss_tolerance_abs,
        )
    if config.strategy == PredictionStrategy.TOTAL_POINTS_REGRESSOR:
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        return TotalPointsStrategy(line_col=config.line_col)
    return LineErrorStrategy(sample_weight=config.sample_weight)
