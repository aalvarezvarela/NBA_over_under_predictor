import numpy as np
import optuna
import pandas as pd

from training_pipeline.config import (
    DataConfig,
    ExperimentConfig,
    OptunaConfig,
    SampleWeightConfig,
    SearchSpaceConfig,
    TargetFamily,
)
from training_pipeline.tuning import (
    LineErrorStrategy,
    TotalPointsStrategy,
    get_strategy,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _total_points_config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "dispatch_test",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="data/train_data/example.csv"),
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _line_error_config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "dispatch_test_le",
        "target_family": TargetFamily.LINE_ERROR,
        "data": DataConfig(csv_path="data/train_data/example.csv"),
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_get_strategy_returns_total_points_strategy_with_configured_line_col():
    config = _total_points_config()
    strategy = get_strategy(config)
    assert isinstance(strategy, TotalPointsStrategy)
    assert strategy.line_col == "ODDS_TOTAL_LINE_bet365"


def test_get_strategy_returns_line_error_strategy_with_configured_sample_weight():
    sw = SampleWeightConfig(enabled=True, lambda_=0.01)
    config = _line_error_config(sample_weight=sw)
    strategy = get_strategy(config)
    assert isinstance(strategy, LineErrorStrategy)
    assert strategy.sample_weight == sw


def _tiny_problem(n: int = 40):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": rng.uniform(200, 240, n),
            "FEATURE_A": rng.normal(size=n),
        }
    )
    y = pd.Series(X["ODDS_TOTAL_LINE_bet365"].to_numpy() + rng.normal(0, 10, n))
    splits = [(np.arange(0, 30), np.arange(30, n))]
    dates = pd.Series(pd.date_range("2026-01-01", periods=n, freq="D"))
    return X, y, splits, dates


def _fast(config_factory, **overrides):
    """One trial, tiny forest -- we are testing wiring, not model quality."""
    return config_factory(
        optuna=OptunaConfig(
            n_trials=1,
            search_space=SearchSpaceConfig(n_estimators=10, early_stopping_rounds=5),
        ),
        **overrides,
    )


def test_total_points_tuning_scores_folds_against_the_configured_line(monkeypatch):
    """The total_points family must score folds with the betting line, so the
    configured line_col has to reach evaluate_fold_total_points.
    """
    import training_pipeline.tuning as tuning_module

    seen: dict = {}
    real = tuning_module._total_points.evaluate_fold_total_points

    def spy(model, X_valid, y_valid, line_col, *, fold, n_train):
        seen["line_col"] = line_col
        return real(model, X_valid, y_valid, line_col, fold=fold, n_train=n_train)

    monkeypatch.setattr(
        tuning_module._total_points, "evaluate_fold_total_points", spy
    )

    X, y, splits, _ = _tiny_problem()
    config = _fast(_total_points_config)
    study = get_strategy(config).tune(X=X, y=y, splits=splits, config=config)

    assert seen["line_col"] == "ODDS_TOTAL_LINE_bet365"
    assert len(study.trials) == 1


def test_line_error_tuning_uses_the_error_fold_evaluator(monkeypatch):
    """The line_error family scores off the sign of the predicted error and
    must never be routed through the line-based evaluator.
    """
    import training_pipeline.tuning as tuning_module

    called = {"error_line": 0, "total_points": 0}
    real = tuning_module._error_line.evaluate_fold_error_line

    def spy(model, X_valid, y_valid, *, fold, n_train):
        called["error_line"] += 1
        return real(model, X_valid, y_valid, fold=fold, n_train=n_train)

    def forbidden(*args, **kwargs):
        called["total_points"] += 1
        raise AssertionError("line_error must not use the total_points evaluator")

    monkeypatch.setattr(tuning_module._error_line, "evaluate_fold_error_line", spy)
    monkeypatch.setattr(
        tuning_module._total_points, "evaluate_fold_total_points", forbidden
    )

    X, y, splits, _ = _tiny_problem()
    config = _fast(_line_error_config)
    get_strategy(config).tune(X=X, y=y, splits=splits, config=config)

    assert called["error_line"] == 1
    assert called["total_points"] == 0


def test_recency_weighting_is_applied_only_when_enabled(monkeypatch):
    import training_pipeline.tuning as tuning_module

    calls = {"n": 0}
    real = tuning_module.build_recency_sample_weights

    def spy(dates, **kwargs):
        calls["n"] += 1
        return real(dates, **kwargs)

    monkeypatch.setattr(tuning_module, "build_recency_sample_weights", spy)
    X, y, splits, dates = _tiny_problem()

    disabled = _fast(_line_error_config)
    get_strategy(disabled).tune(X=X, y=y, splits=splits, config=disabled, dates=dates)
    assert calls["n"] == 0

    enabled = _fast(
        _line_error_config, sample_weight=SampleWeightConfig(enabled=True, lambda_=0.01)
    )
    get_strategy(enabled).tune(X=X, y=y, splits=splits, config=enabled, dates=dates)
    assert calls["n"] == 1


def test_tuned_lambda_is_recorded_on_the_trial():
    """When the decay rate is tuned it becomes part of the trial, so downstream
    refitting can recover it.
    """
    X, y, splits, dates = _tiny_problem()
    config = _fast(
        _line_error_config,
        sample_weight=SampleWeightConfig(
            enabled=True,
            tune_lambda=True,
            lambda_bounds=(1e-4, 0.02),
            # Isolate the always-weighted path; the opt-out is covered in
            # test_training_pipeline_search_space.py.
            allow_unweighted=False,
        ),
    )
    study = get_strategy(config).tune(
        X=X, y=y, splits=splits, config=config, dates=dates
    )

    trial = study.trials[0]
    assert "sample_weight_lambda" in trial.params
    assert 1e-4 <= trial.params["sample_weight_lambda"] <= 0.02
    assert trial.user_attrs["sample_weight_lambda"] == trial.params["sample_weight_lambda"]


def test_objective_records_the_user_attrs_downstream_helpers_read():
    """select_best_trial_lexicographic and summarize_optuna_trials depend on
    these keys; losing one silently breaks trial selection.
    """
    X, y, splits, _ = _tiny_problem()
    config = _fast(_total_points_config)
    study = get_strategy(config).tune(X=X, y=y, splits=splits, config=config)

    attrs = study.trials[0].user_attrs
    for key in (
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "mean_ou_acc",
        "mean_ou_acc_edge_2",
        "mean_best_iteration",
        "median_best_iteration",
        "fold_metrics",
    ):
        assert key in attrs, f"missing user_attr: {key}"
