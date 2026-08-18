import math

import numpy as np
import pandas as pd
import pytest

from training_pipeline.baseline import (
    compute_baseline_metrics,
    compute_baseline_metrics_across_folds,
    compute_baseline_metrics_for_rows,
    compute_bias_corrected_baseline_metrics,
    compute_line_error_bias,
)


def test_baseline_metrics_match_hand_computed_values_for_constant_offset():
    y_true = pd.Series([210.0, 220.0, 200.0, 230.0])
    baseline_line = pd.Series([205.0, 215.0, 195.0, 225.0])  # always 5 points low

    metrics = compute_baseline_metrics(
        y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
    )

    assert metrics.n_games == 4
    assert metrics.mae == pytest.approx(5.0)
    assert metrics.rmse == pytest.approx(5.0)
    assert metrics.r2 == pytest.approx(1.0 - (4 * 25.0) / np.sum((y_true - y_true.mean()) ** 2))


def test_baseline_ou_accuracy_is_nan_not_zero():
    y_true = pd.Series([210.0, 220.0])
    baseline_line = pd.Series([205.0, 215.0])

    metrics = compute_baseline_metrics(
        y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
    )

    assert metrics.ou_accuracy is None


def test_baseline_metrics_drops_non_finite_rows():
    y_true = pd.Series([210.0, np.nan, 200.0])
    baseline_line = pd.Series([205.0, 215.0, np.nan])

    metrics = compute_baseline_metrics(
        y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
    )

    assert metrics.n_games == 1


def test_baseline_metrics_raises_when_no_valid_rows():
    y_true = pd.Series([np.nan, np.nan])
    baseline_line = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError):
        compute_baseline_metrics(
            y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
        )


def test_compute_baseline_metrics_for_rows_slices_by_position():
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [210.0, 220.0, 200.0, 230.0],
            "BASELINE_LINE": [205.0, 215.0, 195.0, 225.0],
        }
    )
    metrics = compute_baseline_metrics_for_rows(
        df, np.array([0, 2]), baseline_line_col="BASELINE_LINE"
    )
    assert metrics.n_games == 2
    assert metrics.mae == pytest.approx(5.0)


def test_compute_line_error_bias_is_mean_signed_error():
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [215.0, 225.0, 205.0],
            "LINE": [210.0, 210.0, 210.0],
        }
    )
    assert compute_line_error_bias(df, baseline_line_col="LINE") == pytest.approx(5.0)


def test_compute_line_error_bias_rejects_all_missing_rows():
    df = pd.DataFrame({"TOTAL_POINTS": [np.nan], "LINE": [210.0]})
    with pytest.raises(ValueError):
        compute_line_error_bias(df, baseline_line_col="LINE")


def test_bias_corrected_baseline_is_a_harder_null_than_the_raw_line():
    """Shifting the line by its historical drift must reduce MAE whenever the
    line is systematically biased -- that is the point of the stronger null.
    """
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [215.0, 216.0, 214.0, 215.0],
            "LINE": [210.0, 210.0, 210.0, 210.0],
        }
    )
    raw = compute_baseline_metrics(
        y_true_total_points=df["TOTAL_POINTS"],
        baseline_line=df["LINE"],
        line_col="LINE",
    )
    bias = compute_line_error_bias(df, baseline_line_col="LINE")
    corrected = compute_bias_corrected_baseline_metrics(
        df, baseline_line_col="LINE", bias=bias
    )

    assert bias == pytest.approx(5.0)
    assert raw.mae == pytest.approx(5.0)
    assert corrected.mae == pytest.approx(0.5)
    assert corrected.mae < raw.mae
    assert "bias" in corrected.line_col


def test_compute_baseline_metrics_across_folds_uses_validation_rows_only():
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [210.0, 220.0, 200.0, 230.0, 240.0, 250.0],
            "BASELINE_LINE": [205.0, 215.0, 195.0, 220.0, 230.0, 240.0],
        }
    )
    splits = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]

    fold_df, aggregate = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE"
    )

    assert len(fold_df) == 2
    assert fold_df.loc[0, "mae"] == pytest.approx(7.5)  # |200-195|=5, |230-220|=10 -> mean 7.5
    assert fold_df.loc[1, "mae"] == pytest.approx(10.0)  # |240-230|=10, |250-240|=10
    assert aggregate.mae == pytest.approx((7.5 + 10.0) / 2)
    assert aggregate.ou_accuracy is None
    assert not math.isnan(aggregate.mae)
