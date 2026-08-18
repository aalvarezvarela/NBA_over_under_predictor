from pathlib import Path

import pytest
import yaml

from training_pipeline.cli import deep_merge, find_base_config, load_config

REPO_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def test_deep_merge_overrides_leaves_without_dropping_siblings():
    base = {"optuna": {"n_trials": 80, "objective_name": "reg:squarederror"}}
    override = {"optuna": {"n_trials": 5}}
    assert deep_merge(base, override) == {
        "optuna": {"n_trials": 5, "objective_name": "reg:squarederror"}
    }


def test_deep_merge_replaces_lists_wholesale():
    """Appending would make it impossible to shrink a list such as
    edge_thresholds.
    """
    base = {"betting": {"edge_thresholds": [0.0, 1.0, 2.0]}}
    override = {"betting": {"edge_thresholds": [3.0]}}
    assert deep_merge(base, override)["betting"]["edge_thresholds"] == [3.0]


def test_deep_merge_is_recursive():
    base = {"optuna": {"search_space": {"max_depth": {"low": 2, "high": 4}}}}
    override = {"optuna": {"search_space": {"max_depth": {"high": 9}}}}
    merged = deep_merge(base, override)
    assert merged["optuna"]["search_space"]["max_depth"] == {"low": 2, "high": 9}


def _write(tmp_path: Path, rel: str, payload: dict) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))
    return path


def test_experiment_inherits_base_from_a_parent_directory(tmp_path):
    _write(
        tmp_path,
        "_base.yaml",
        {
            "data": {"csv_path": "x.csv", "season_year_floor": 2021},
            "optuna": {"n_trials": 80, "objective_name": "reg:squarederror"},
        },
    )
    experiment = _write(
        tmp_path,
        "total_points/run.yaml",
        {
            "experiment_name": "e",
            "target_family": "total_points",
            "line_col": "TOTAL_LINE_bet365",
            "optuna": {"n_trials": 3},
        },
    )

    config = load_config(experiment)

    assert config.optuna.n_trials == 3                     # overridden
    assert config.optuna.objective_name == "reg:squarederror"  # inherited
    assert config.data.season_year_floor == 2021           # inherited
    assert config.experiment_name == "e"


def test_use_base_false_loads_the_file_in_isolation(tmp_path):
    _write(tmp_path, "_base.yaml", {"data": {"season_year_floor": 2021}})
    experiment = _write(
        tmp_path,
        "run.yaml",
        {
            "experiment_name": "e",
            "target_family": "total_points",
            "line_col": "TOTAL_LINE_bet365",
            "data": {"csv_path": "x.csv"},
        },
    )

    assert load_config(experiment, use_base=False).data.season_year_floor is None
    assert load_config(experiment).data.season_year_floor == 2021


def test_find_base_config_returns_none_when_absent(tmp_path):
    experiment = _write(tmp_path, "run.yaml", {"experiment_name": "e"})
    assert find_base_config(experiment) is None


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(TypeError, match="mapping"):
        load_config(path, use_base=False)


# --- the checked-in experiment definitions must actually be valid -----------


@pytest.mark.parametrize(
    "relative_path",
    ["total_points/3_seasons.yaml", "line_error/3_seasons_weighted.yaml"],
)
def test_repo_experiment_definitions_are_valid(relative_path):
    config = load_config(REPO_EXPERIMENTS / relative_path)
    assert config.experiment_name
    assert config.comparison_group == "window_sweep_2026_03"
    # Inherited from _base.yaml.
    assert config.data.season_year_floor == 2021
    assert config.data.exclude_playoffs is True
    # Compare to the code defaults rather than a literal, so this test
    # tracks intentional changes instead of blocking them.
    assert config.optuna.search_space == expected_search_space(config)


def test_repo_line_error_definition_enables_recency_weighting():
    config = load_config(REPO_EXPERIMENTS / "line_error/3_seasons_weighted.yaml")
    assert config.target_family == "line_error"
    assert config.line_col is None
    assert config.sample_weight.enabled is True
    assert config.sample_weight.tune_lambda is True
    assert config.optuna.mae_tolerance_abs == 0.05  # overrides base 0.10


def expected_search_space(config):
    """The space a config should resolve to, given the defaults it inherits.

    Two automatic transforms are applied by ExperimentConfig validation and both
    are intentional, so a test comparing against a bare ``SearchSpaceConfig()``
    would now fail on the transform rather than on drift:

      * a classifier inherits CLASSIFIER_SEARCH_SPACE (hessian-scaled ranges);
      * ``optuna.tune_n_estimators`` -- on by default in _base.yaml -- fills in
        the strategy's n_estimators range from N_ESTIMATORS_RANGES.

    Rebuilding the expectation from those same sources keeps the guard: change
    any OTHER range in _base.yaml and this still fails.
    """
    from training_pipeline.config import (
        CLASSIFIER_SEARCH_SPACE,
        N_ESTIMATORS_RANGES,
        SearchSpaceConfig,
    )

    base = (
        CLASSIFIER_SEARCH_SPACE if config.is_classifier else SearchSpaceConfig()
    ).model_copy(deep=True)
    if config.optuna.tune_n_estimators:
        base.n_estimators_range = N_ESTIMATORS_RANGES[config.strategy].model_copy(
            deep=True
        )
    return base


def test_base_yaml_search_space_matches_the_code_defaults():
    """_base.yaml documents the search space; if it drifts from the pydantic
    defaults the file becomes misleading.
    """
    config = load_config(REPO_EXPERIMENTS / "total_points/3_seasons.yaml")
    assert config.optuna.search_space == expected_search_space(config)


# --- the checked-in campaign must be correct as written ---------------------

CAMPAIGN = REPO_EXPERIMENTS / "strategy_window_2026_08"


@pytest.mark.parametrize(
    "name",
    [
        "total_points_2500", "total_points_3750",
        "line_error_2500", "line_error_3750",
        "classifier_2500", "classifier_3750",
    ],
)
def test_campaign_configs_are_valid_and_share_their_controls(name):
    """Everything except strategy and window is held fixed. A six-cell
    comparison is only readable if one thing varies per axis.
    """
    config = load_config(CAMPAIGN / f"{name}.yaml")

    assert config.comparison_group == "strategy_window_2026_08"
    assert config.optuna.n_trials == 50
    # Fixed trial count, no timeout: the previous A/B gave one side 2.4x the
    # tuning because its timeout was larger, and a timeout also makes the trial
    # count a function of the machine.
    assert config.optuna.timeout is None
    assert config.holdout.test_days == 60
    assert config.evaluation_seeds == (101, 202)
    assert config.data.exclude_overtime_from_training is False


def test_campaign_classifiers_get_the_hessian_scaled_space():
    """The regression space's upper half drives a classifier to a constant
    prediction, so inheriting it would make both classifier runs uninformative.
    """
    from training_pipeline.config import CLASSIFIER_SEARCH_SPACE

    for window in (2500, 3750):
        config = load_config(CAMPAIGN / f"classifier_{window}.yaml")
        assert config.optuna.search_space == expected_search_space(config), window
        # The hessian-scaled ranges are what matters here; assert one directly
        # so the test still fails if the swap stops firing.
        assert config.optuna.search_space.gamma == CLASSIFIER_SEARCH_SPACE.gamma
        assert config.optuna.objective_name == "binary:logistic"


def test_campaign_regressors_keep_the_regression_space():
    from training_pipeline.config import SearchSpaceConfig

    for name in ("total_points_2500", "line_error_3750"):
        config = load_config(CAMPAIGN / f"{name}.yaml")
        assert config.optuna.search_space == expected_search_space(config), name
        assert config.optuna.search_space.gamma == SearchSpaceConfig().gamma, name
        assert config.optuna.objective_name == "reg:squarederror"


def test_campaign_line_col_matches_each_strategy():
    """line_error must NOT carry one (its target is already relative to the
    line); the other two must, and for the classifier it defines the label.
    """
    assert load_config(CAMPAIGN / "line_error_2500.yaml").line_col is None
    assert load_config(CAMPAIGN / "total_points_2500.yaml").line_col == "TOTAL_LINE_bet365"
    assert load_config(CAMPAIGN / "classifier_2500.yaml").line_col == "TOTAL_LINE_bet365"


def test_campaign_cells_are_pairwise_distinct_studies():
    """Six configs, six fingerprints -- otherwise two cells could share a
    persistent Optuna study and silently pool incomparable trials.
    """
    names = [
        "total_points_2500", "total_points_3750",
        "line_error_2500", "line_error_3750",
        "classifier_2500", "classifier_3750",
    ]
    fingerprints = {load_config(CAMPAIGN / f"{n}.yaml").fingerprint() for n in names}
    assert len(fingerprints) == len(names)


# --- seven-snapshot intermediate-line campaign -----------------------------

INTERMEDIATE_CAMPAIGN = REPO_EXPERIMENTS / "intermediate_line_2026_08"
INTERMEDIATE_COMPARISON_GROUP = "intermediate_line_7snapshot_6h_4h_2026_08"

# Six, not the two the rest of the repo uses: the snapshot cells are separated
# by fractions of a point, so the seed spread has to be narrow enough to tell
# them apart. Every `_no_time_decay` config in this campaign sets the same six,
# and the comparison is only meaningful if they all agree.
INTERMEDIATE_EVALUATION_SEEDS = (101, 202, 303, 404, 505, 606)


@pytest.mark.parametrize(
    "name",
    [
        "pooled_7snapshot_line_error_no_time_decay",
        "t360_control_line_error_no_time_decay",
        "t240_control_line_error_no_time_decay",
    ],
)
def test_intermediate_7snapshot_campaign_is_line_error_only_and_unweighted(name):
    config = load_config(INTERMEDIATE_CAMPAIGN / f"{name}.yaml")

    assert config.comparison_group == INTERMEDIATE_COMPARISON_GROUP
    assert config.target_family == "line_error"
    assert config.line_col is None
    assert config.sample_weight.enabled is False
    assert config.sample_weight.lambda_ is None
    assert config.sample_weight.tune_lambda is False
    assert config.optuna.n_trials == 50
    assert config.optuna.timeout is None
    assert config.holdout.test_days == 60
    assert config.evaluation_seeds == INTERMEDIATE_EVALUATION_SEEDS


def test_intermediate_7snapshot_pooled_window_is_scaled_by_seven():
    config = load_config(
        INTERMEDIATE_CAMPAIGN / "pooled_7snapshot_line_error_no_time_decay.yaml"
    )

    assert Path(config.data.csv_path).name == "intermediate_line_data_20260412_7snap.csv"
    assert config.walk_forward.train_games == 26_250
    assert config.walk_forward.min_train_games == 13_125
    assert config.walk_forward.test_games == 350
    assert config.walk_forward.step_games_between_tests == 420
    assert config.backtest.test_games == 2_100


@pytest.mark.parametrize("horizon", [360, 240])
def test_intermediate_single_snapshot_controls_keep_unscaled_windows(horizon):
    config = load_config(
        INTERMEDIATE_CAMPAIGN
        / f"t{horizon}_control_line_error_no_time_decay.yaml"
    )

    assert Path(config.data.csv_path).name.endswith(f"_7snap_t{horizon}.csv")
    assert config.walk_forward.train_games == 3_750
    assert config.walk_forward.min_train_games == 1_875
    assert config.walk_forward.test_games == 50
    assert config.walk_forward.step_games_between_tests == 60
    assert config.backtest.test_games == 300
