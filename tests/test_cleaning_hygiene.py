"""Tests for the cleaning pipeline's hygiene fixes and the cleaning report.

Each of these pins a bug that was found by audit rather than by a failure --
none of them made anything crash, which is why they lasted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    MIN_PLAUSIBLE_TOTAL_LINE,
    MIN_PLAUSIBLE_TOTAL_POINTS,
    _normalize_nullable_dtypes,
    advanced_column_cleaning,
    basic_cleaning,
    clean_dataframe_for_training,
)
from nba_ou.data_processing.missing_data.cleaning_report import CleaningReport


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(16)
    n = 60
    return pd.DataFrame(
        {
            "GAME_ID": [f"002240000{i:02d}" for i in range(n)],
            "TOTAL_POINTS": rng.normal(225, 20, n).round(),
            "ODDS_TOTAL_LINE_bet365": rng.normal(224, 15, n).round(),
            "ODDS_SPREAD_bet365": rng.normal(0, 6, n).round(),
            "ODDS_MONEYLINE_bet365_TEAM_HOME": rng.normal(-110, 40, n).round(),
            "ODDS_MONEYLINE_bet365_TEAM_AWAY": rng.normal(-110, 40, n).round(),
            "IS_US_HOLIDAY_BEFORE": [i % 7 == 0 for i in range(n)],
            "PACE_BEFORE_TEAM_HOME": rng.normal(100, 4, n),
        }
    )


# ---------------------------------------------------------------------------
# input mutation
# ---------------------------------------------------------------------------


def test_basic_cleaning_does_not_mutate_its_input(frame):
    """The dtype casts used to land on the caller's frame, so cleaning the same
    dataframe twice did not do the same thing twice."""
    before = frame.copy(deep=True)
    basic_cleaning(frame, verbose=0)

    pd.testing.assert_frame_equal(frame, before)


def test_clean_dataframe_for_training_does_not_mutate_its_input(frame):
    before = frame.copy(deep=True)
    clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)

    pd.testing.assert_frame_equal(frame, before)


def test_cleaning_is_idempotent_across_two_calls(frame):
    """The observable consequence of the mutation bug."""
    first = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)
    second = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)

    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# the implausible-total filter
# ---------------------------------------------------------------------------


def test_implausibly_low_totals_are_dropped(frame):
    frame.loc[0, "TOTAL_POINTS"] = 12.0
    frame.loc[1, "TOTAL_POINTS"] = float(MIN_PLAUSIBLE_TOTAL_POINTS)

    cleaned = basic_cleaning(frame, verbose=0)

    assert len(cleaned) == len(frame) - 2
    assert (cleaned["TOTAL_POINTS"] > MIN_PLAUSIBLE_TOTAL_POINTS).all()


def test_unknown_totals_survive(frame):
    """The prediction path cleans scheduled games, which have no final score.
    A plain `df[df.TOTAL_POINTS > 130]` compares against NaN, is False for every
    such row, and would silently return an empty frame -- deleting the daily
    prediction run rather than failing it."""
    frame["TOTAL_POINTS"] = np.nan

    cleaned = basic_cleaning(frame, verbose=0)

    assert len(cleaned) == len(frame)


def test_a_frame_with_no_target_column_still_cleans(frame):
    cleaned = basic_cleaning(frame.drop(columns=["TOTAL_POINTS"]), verbose=0)
    assert len(cleaned) == len(frame)


def test_implausible_lines_are_dropped(frame):
    frame.loc[0, "ODDS_TOTAL_LINE_bet365"] = float(MIN_PLAUSIBLE_TOTAL_LINE)
    cleaned = basic_cleaning(frame, verbose=0)
    assert len(cleaned) == len(frame) - 1


# ---------------------------------------------------------------------------
# nullable dtypes
# ---------------------------------------------------------------------------


def test_nullable_boolean_with_na_becomes_float_not_object():
    """A nullable boolean holding pd.NA converts via to_numpy() to dtype object,
    which XGBoost rejects. The previous normalisation could not reach it:
    select_dtypes(include=[np.number]) does not match `boolean`."""
    df = pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean")})

    assert df["flag"].to_numpy().dtype == object  # the trap

    out = _normalize_nullable_dtypes(df)

    assert out["flag"].dtype == np.float64
    assert out["flag"].to_numpy().dtype == np.float64
    assert np.isnan(out["flag"].to_numpy()[2])


def test_holiday_flag_leaves_cleaning_as_a_numpy_dtype(frame):
    """basic_cleaning deliberately casts this column to nullable boolean, so the
    end of the pipeline has to undo it."""
    frame["IS_US_HOLIDAY_BEFORE"] = frame["IS_US_HOLIDAY_BEFORE"].astype(object)
    frame.loc[0, "IS_US_HOLIDAY_BEFORE"] = None

    cleaned = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)

    assert not isinstance(
        cleaned["IS_US_HOLIDAY_BEFORE"].dtype, pd.api.extensions.ExtensionDtype
    )
    assert cleaned["IS_US_HOLIDAY_BEFORE"].to_numpy().dtype != object


def test_nullable_integer_is_normalised():
    df = pd.DataFrame({"n": pd.array([1, 2, None], dtype="Int64")})
    out = _normalize_nullable_dtypes(df)
    assert out["n"].dtype == np.float64


def test_categorical_columns_are_left_alone():
    """XGBoost consumes pandas Categorical natively under enable_categorical,
    which is what categorical_team_encoding relies on."""
    df = pd.DataFrame({"team": pd.Categorical(["BOS", "LAL", "BOS"])})
    out = _normalize_nullable_dtypes(df)
    assert isinstance(out["team"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------------------
# empty frame
# ---------------------------------------------------------------------------


def test_empty_frame_does_not_warn_or_drop_every_column(frame):
    """0/0 is not 100% NaN. Without the guard this emitted a RuntimeWarning and
    then dropped nothing anyway, because every comparison against NaN is False."""
    empty = frame.iloc[:0]

    with warnings_as_errors():
        cleaned = advanced_column_cleaning(empty, nan_threshold=50.0, verbose=0)

    assert len(cleaned.columns) > 0


def warnings_as_errors():
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            yield

    return ctx()


# ---------------------------------------------------------------------------
# the cleaning report
# ---------------------------------------------------------------------------


def test_report_is_only_returned_when_asked(frame):
    plain = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)
    assert isinstance(plain, pd.DataFrame)

    pair = clean_dataframe_for_training(
        frame, verbose=0, keep_all_cols=True, return_report=True
    )
    assert isinstance(pair, tuple) and isinstance(pair[1], CleaningReport)


def test_report_names_the_step_that_dropped_a_column(frame):
    frame["ALWAYS_SAME"] = 1.0
    frame["A_STRING_COL"] = "text"

    _, report = clean_dataframe_for_training(
        frame, nan_threshold=50.0, verbose=0, return_report=True
    )

    assert report.why_dropped("ALWAYS_SAME")["step"] == "constant_columns"
    assert report.why_dropped("A_STRING_COL")["step"] == "string_columns"
    assert report.why_dropped("PACE_BEFORE_TEAM_HOME") is None


def test_report_records_the_correlation_partner(frame):
    frame["PACE_COPY_BEFORE"] = frame["PACE_BEFORE_TEAM_HOME"] * 2.0

    _, report = clean_dataframe_for_training(
        frame,
        corr_threshold=0.95,
        corr_threshold_overrides={},
        verbose=0,
        return_report=True,
    )

    dropped = [
        e
        for e in report.column_drops
        if e["step"] in ("correlated_columns", "duplicate_columns")
    ]
    assert dropped, "the duplicated column should have been recorded"
    assert "PACE" in dropped[0]["reason"] or "r=" in dropped[0]["reason"]


def test_report_records_row_drops_with_reasons(frame):
    frame.loc[0, "TOTAL_POINTS"] = 10.0
    frame.loc[1, "ODDS_TOTAL_LINE_bet365"] = np.nan

    _, report = clean_dataframe_for_training(frame, verbose=0, return_report=True)

    steps = {entry["step"] for entry in report.row_drops}
    assert "basic_cleaning.implausible_total" in steps
    assert "basic_cleaning.missing_total_line" in steps
    assert all(entry["rows_dropped"] > 0 for entry in report.row_drops)


def test_report_totals_match_the_returned_frame(frame):
    frame["A_STRING_COL"] = "text"
    cleaned, report = clean_dataframe_for_training(
        frame, nan_threshold=50.0, verbose=0, return_report=True
    )

    assert report.columns_in == len(frame.columns)
    assert report.columns_out == len(cleaned.columns)
    assert report.rows_in == len(frame)
    assert report.rows_out == len(cleaned)
    # Every dropped column is accounted for exactly once, so the per-step counts
    # sum to the columns that actually went. GAME_ID satisfies both the
    # pure-string and the _ID rule and must still be reported once.
    assert len(report.column_drops) == report.columns_in - report.columns_out
    recorded = [entry["column"] for entry in report.column_drops]
    assert len(recorded) == len(set(recorded))
    assert sum(report.columns_by_step().values()) == len(report.column_drops)


def test_report_serialises_to_json(frame, tmp_path):
    frame["A_STRING_COL"] = "text"
    _, report = clean_dataframe_for_training(frame, verbose=0, return_report=True)

    path = report.save(tmp_path / "cleaning_report.json")

    import json

    payload = json.loads(path.read_text())
    assert payload["columns_in"] == len(frame.columns)
    assert "column_drops" in payload and "columns_dropped_by_step" in payload
