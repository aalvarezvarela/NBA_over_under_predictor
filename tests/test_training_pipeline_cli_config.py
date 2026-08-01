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
    assert config.optuna.search_space.max_depth.low == 2


def test_repo_line_error_definition_enables_recency_weighting():
    config = load_config(REPO_EXPERIMENTS / "line_error/3_seasons_weighted.yaml")
    assert config.target_family == "line_error"
    assert config.line_col is None
    assert config.sample_weight.enabled is True
    assert config.sample_weight.tune_lambda is True
    assert config.optuna.mae_tolerance_abs == 0.05  # overrides base 0.10


def test_base_yaml_search_space_matches_the_code_defaults():
    """_base.yaml documents the search space; if it drifts from the pydantic
    defaults the file becomes misleading.
    """
    from training_pipeline.config import SearchSpaceConfig

    config = load_config(REPO_EXPERIMENTS / "total_points/3_seasons.yaml")
    assert config.optuna.search_space == SearchSpaceConfig()
