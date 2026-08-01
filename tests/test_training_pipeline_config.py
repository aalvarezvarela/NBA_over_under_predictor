import pytest
from pydantic import ValidationError

from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    OptunaConfig,
    RefitConfig,
    SampleWeightConfig,
    TargetFamily,
    WalkForwardConfig,
)


def _base_kwargs(**overrides):
    kwargs = {
        "experiment_name": "test_experiment",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="data/train_data/example.csv"),
    }
    kwargs.update(overrides)
    return kwargs


def test_total_points_requires_line_col():
    with pytest.raises(ValidationError, match="line_col is required"):
        ExperimentConfig(**_base_kwargs(line_col=None))


def test_line_error_forbids_line_col():
    kwargs = _base_kwargs(target_family=TargetFamily.LINE_ERROR, line_col="TOTAL_LINE_bet365")
    with pytest.raises(ValidationError, match="line_col must be omitted"):
        ExperimentConfig(**kwargs)


def test_total_points_accepts_recency_weighting():
    """Previously rejected, because upstream's optuna_total_points.py has no
    sample-weight parameters. training_pipeline supplies its own objective and
    final-fit path, so both target families support weighting.
    """
    config = ExperimentConfig(
        **_base_kwargs(sample_weight=SampleWeightConfig(enabled=True, lambda_=0.005))
    )
    assert config.sample_weight.enabled is True
    assert config.sample_weight.lambda_ == 0.005


def test_total_points_accepts_tuned_sample_weight_lambda():
    config = ExperimentConfig(
        **_base_kwargs(
            sample_weight=SampleWeightConfig(
                enabled=True, tune_lambda=True, lambda_bounds=(1e-4, 0.02)
            )
        )
    )
    assert config.sample_weight.tune_lambda is True


def test_sample_weighting_changes_the_fingerprint():
    """Weighted and unweighted training are different problems, so a
    persistent study must not be resumed across them.
    """
    unweighted = ExperimentConfig(**_base_kwargs()).fingerprint()
    weighted = ExperimentConfig(
        **_base_kwargs(sample_weight=SampleWeightConfig(enabled=True, lambda_=0.005))
    ).fingerprint()
    assert unweighted != weighted


def test_line_error_auto_adds_required_exclude_cols():
    kwargs = _base_kwargs(target_family=TargetFamily.LINE_ERROR, line_col=None)
    config = ExperimentConfig(**kwargs)
    assert "LINE_ERROR" in config.exclude_cols
    assert "TOTAL_POINTS" in config.exclude_cols


def test_total_points_auto_adds_total_points_exclude_col():
    kwargs = _base_kwargs(exclude_cols=["SEASON_YEAR", "GAME_DATE"])
    config = ExperimentConfig(**kwargs)
    assert "TOTAL_POINTS" in config.exclude_cols


def test_config_round_trips_through_json():
    config = ExperimentConfig(**_base_kwargs())
    restored = ExperimentConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_refit_has_no_separate_train_games_knob():
    """Regression: the CV window and the final-refit window must be a single
    value. Two knobs could disagree, which would select hyperparameters for one
    training-set size and then fit the shipped model on another.
    """
    assert "train_games" not in RefitConfig.model_fields

    config = ExperimentConfig(
        **_base_kwargs(walk_forward=WalkForwardConfig(train_games=1234))
    )
    assert config.walk_forward.train_games == 1234
    assert config.resolved_window_dir_label == "1234_games"


def test_legacy_refit_train_games_in_yaml_is_ignored_not_fatal():
    """Configs written before the unification must still load."""
    config = ExperimentConfig.model_validate(
        {
            **_base_kwargs(),
            "refit": {"strategy": "rolling_window", "train_games": 999},
        }
    )
    assert not hasattr(config.refit, "train_games")


def test_training_version_defaults_to_unset():
    assert ExperimentConfig(**_base_kwargs()).training_version is None


def test_training_version_is_kept_verbatim_when_set():
    config = ExperimentConfig(**_base_kwargs(training_version="2.1-style-features"))
    assert config.training_version == "2.1-style-features"


def test_blank_training_version_is_treated_as_unset():
    """A whitespace-only label is a mistake, not a version."""
    assert ExperimentConfig(**_base_kwargs(training_version="   ")).training_version is None
    assert ExperimentConfig(**_base_kwargs(training_version="")).training_version is None


def test_training_version_is_stripped():
    config = ExperimentConfig(**_base_kwargs(training_version="  v3  "))
    assert config.training_version == "v3"


def test_training_version_does_not_change_the_fingerprint():
    """It is a human label: relabelling must not fork a persistent study, and
    real changes are already captured by the fingerprint on their own.
    """
    base = ExperimentConfig(**_base_kwargs()).fingerprint()
    labelled = ExperimentConfig(**_base_kwargs(training_version="2.1")).fingerprint()
    relabelled = ExperimentConfig(**_base_kwargs(training_version="9.9")).fingerprint()

    assert base == labelled == relabelled


def test_training_version_round_trips_through_json():
    config = ExperimentConfig(**_base_kwargs(training_version="2.1"))
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config


def test_playoffs_are_excluded_by_default():
    config = ExperimentConfig(**_base_kwargs())
    assert config.data.exclude_playoffs is True
    assert config.data.allowed_season_types == ("Regular Season", "Play-In Tournament")
    assert "Playoffs" not in config.data.allowed_season_types


def test_allowed_season_types_must_be_known_values():
    with pytest.raises(ValidationError, match="Unknown season types"):
        ExperimentConfig(
            **_base_kwargs(
                data=DataConfig(
                    csv_path="x.csv", allowed_season_types=("Regular Season", "Postseason")
                )
            )
        )


def test_allowed_season_types_cannot_be_empty_while_filtering():
    with pytest.raises(ValidationError, match="must not be empty"):
        ExperimentConfig(
            **_base_kwargs(
                data=DataConfig(csv_path="x.csv", exclude_playoffs=True, allowed_season_types=())
            )
        )


def test_toggling_playoff_exclusion_changes_the_fingerprint():
    """Including playoffs changes the training distribution, so a persistent
    Optuna study must not be resumed across the two settings.
    """
    excluded = ExperimentConfig(**_base_kwargs()).fingerprint()
    included = ExperimentConfig(
        **_base_kwargs(
            data=DataConfig(
                csv_path="data/train_data/example.csv",
                allowed_season_types=("Regular Season", "Play-In Tournament", "Playoffs"),
            )
        )
    ).fingerprint()
    assert excluded != included


def test_fingerprint_ignores_cosmetic_and_output_only_fields():
    """Renaming an experiment or asking for more trials must still resume the
    same persistent Optuna study -- those don't change what a trial means.
    """
    base = ExperimentConfig(**_base_kwargs()).fingerprint()

    assert ExperimentConfig(**_base_kwargs(experiment_name="renamed")).fingerprint() == base
    assert ExperimentConfig(**_base_kwargs(window_dir_label="3_seasons")).fingerprint() == base
    assert (
        ExperimentConfig(**_base_kwargs(optuna=OptunaConfig(n_trials=500))).fingerprint()
        == base
    )
    assert (
        ExperimentConfig(**_base_kwargs(save_experiment_artifacts=False)).fingerprint()
        == base
    )


def test_fingerprint_changes_when_trials_would_not_be_comparable():
    """Regression: persistent Optuna storage used to be keyed only on the
    experiment name, so changing the CSV or fold layout silently appended
    incomparable trials to an existing study.
    """
    base = ExperimentConfig(**_base_kwargs()).fingerprint()

    different_csv = _base_kwargs(data=DataConfig(csv_path="data/train_data/other.csv"))
    assert ExperimentConfig(**different_csv).fingerprint() != base

    different_cleaning = _base_kwargs(cleaning=CleaningConfig(nan_threshold=50.0))
    assert ExperimentConfig(**different_cleaning).fingerprint() != base

    different_window = _base_kwargs(walk_forward=WalkForwardConfig(train_games=2500))
    assert ExperimentConfig(**different_window).fingerprint() != base

    different_objective = _base_kwargs(
        optuna=OptunaConfig(objective_name="reg:pseudohubererror")
    )
    assert ExperimentConfig(**different_objective).fingerprint() != base


def test_fingerprint_is_stable_across_equivalent_constructions():
    assert (
        ExperimentConfig(**_base_kwargs()).fingerprint()
        == ExperimentConfig(**_base_kwargs()).fingerprint()
    )


def test_line_error_config_round_trips_through_json():
    kwargs = _base_kwargs(
        target_family=TargetFamily.LINE_ERROR,
        line_col=None,
        sample_weight=SampleWeightConfig(enabled=True, tune_lambda=True, lambda_=0.005),
    )
    config = ExperimentConfig(**kwargs)
    restored = ExperimentConfig.model_validate_json(config.model_dump_json())
    assert restored == config
