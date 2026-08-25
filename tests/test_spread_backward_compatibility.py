"""Adding the spread market must not disturb any existing totals run.

The specific hazard: ``ExperimentConfig.exclude_cols`` is INSIDE
``fingerprint()``, and the fingerprint keys persistent Optuna storage. Appending
the new outcome columns to ``exclude_cols`` -- the obvious way to block them from
the feature matrix -- would therefore have changed the fingerprint of every
archived totals config and forked every study, silently: a forked study just
starts from zero trials and the run still completes.

They are blocked centrally in ``prepare_dataset`` instead. These tests are what
keep that decision from being quietly undone.
"""

from __future__ import annotations

import pytest

from training_pipeline.cli import load_config
from training_pipeline.config import (
    LEAKING_TARGET_COLUMNS,
    OUTCOME_ONLY_COLUMNS,
    DataConfig,
    ExperimentConfig,
    Market,
    PredictionStrategy,
    TargetFamily,
)

#: Fingerprints measured on the commit BEFORE spread support was added, for a
#: spread of campaigns covering both strategies and both dataset types. If one of
#: these changes, an existing Optuna study has been forked.
PINNED_FINGERPRINTS = {
    "experiments/rolling_origin_2026_08/line_error.yaml": "5e7b9b28fbd3",
    "experiments/rolling_origin_2026_08/total_points.yaml": "cd4f2e2a3ce5",
    "experiments/target_line_error_2026_08/a_closing_reference.yaml": "5e7b9b28fbd3",
    "experiments/target_line_error_2026_08/b_intermediate_pooled.yaml": "c4e3dd202aef",
    "experiments/target_total_points_2026_08/a_closing_reference.yaml": "cd4f2e2a3ce5",
    "experiments/target_total_points_2026_08/b_intermediate_pooled.yaml": "7a26c63f6606",
}


@pytest.mark.parametrize(
    ("config_path", "expected"), sorted(PINNED_FINGERPRINTS.items())
)
def test_archived_totals_fingerprints_are_unchanged(config_path, expected):
    assert load_config(config_path).fingerprint() == expected


def test_outcome_columns_are_not_in_exclude_cols():
    """The mechanism, not just the symptom.

    If someone 'tidies up' by moving OUTCOME_ONLY_COLUMNS into the validator that
    appends LEAKING_TARGET_COLUMNS to exclude_cols, every fingerprint above
    changes. This says why, at the point where it would happen.
    """
    config = ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(csv_path="x.csv"),
        save_experiment_artifacts=False,
    )
    for column in OUTCOME_ONLY_COLUMNS:
        assert column not in config.exclude_cols, (
            f"{column} reached exclude_cols, which is inside fingerprint(). "
            "Every archived totals study would be forked."
        )


def test_leaking_target_columns_are_unchanged():
    """The historical list must keep exactly its historical contents."""
    assert LEAKING_TARGET_COLUMNS == (
        "TOTAL_POINTS",
        "LINE_ERROR",
        "OVER_LABEL",
        "DIFF_FROM_LINE",
        "IS_OVERTIME",
    )


def test_existing_strategy_values_are_unchanged():
    """Legacy configs name these as strings; renaming one breaks every load."""
    assert PredictionStrategy.TOTAL_POINTS_REGRESSOR.value == "total_points_regressor"
    assert PredictionStrategy.LINE_ERROR_REGRESSOR.value == "line_error_regressor"
    assert PredictionStrategy.OVER_UNDER_CLASSIFIER.value == "over_under_classifier"
    assert TargetFamily.TOTAL_POINTS.value == "total_points"
    assert TargetFamily.LINE_ERROR.value == "line_error"
    assert TargetFamily.OVER_UNDER.value == "over_under"


def test_legacy_target_family_only_configs_still_load():
    """Configs predating prediction_strategy set target_family alone."""
    config = ExperimentConfig(
        experiment_name="legacy",
        target_family=TargetFamily.LINE_ERROR,
        data=DataConfig(csv_path="x.csv"),
        save_experiment_artifacts=False,
    )
    assert config.strategy is PredictionStrategy.LINE_ERROR_REGRESSOR
    assert config.market is Market.TOTALS


def test_every_totals_strategy_still_reports_the_totals_market():
    for strategy in (
        PredictionStrategy.TOTAL_POINTS_REGRESSOR,
        PredictionStrategy.LINE_ERROR_REGRESSOR,
        PredictionStrategy.OVER_UNDER_CLASSIFIER,
    ):
        assert strategy.market is Market.TOTALS


def test_spread_is_the_only_new_strategy():
    """Scope guard: no margin regressor, no cover classifier, no moneyline."""
    assert {s.value for s in PredictionStrategy} == {
        "total_points_regressor",
        "line_error_regressor",
        "over_under_classifier",
        "spread_error_regressor",
    }


def test_moneyline_market_exists_but_has_no_strategy():
    """Data readiness only -- deliberately no model."""
    assert Market.MONEYLINE.value == "moneyline"
    assert all(s.market is not Market.MONEYLINE for s in PredictionStrategy)
