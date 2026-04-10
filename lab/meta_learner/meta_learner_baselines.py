from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col

DEFAULT_TOTAL_POINTS_MODEL_COLS = [
    "PRED_TOTAL_POINTS_FULL_DATASET",
    "PRED_TOTAL_POINTS_LAST_5_SEASONS",
    "PRED_TOTAL_POINTS_LAST_3_SEASONS",
]
DEFAULT_LINE_ERROR_MODEL_COLS = [
    "PRED_LINE_ERROR_FULL_DATASET",
    "PRED_LINE_ERROR_LAST_5_SEASONS",
    "PRED_LINE_ERROR_LAST_3_SEASONS",
]
DEFAULT_REFERENCE_TIEBREAKER_COL = "PRED_LINE_ERROR_FULL_DATASET"
BASE_AVG_ALL_6_ERR_COL = "BASE_AVG_ALL_6_ERR"
BASE_MAJORITY_TOTAL_ONLY_ERR_COL = "BASE_MAJORITY_TOTAL_ONLY_ERR"
BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL = "BASE_MAJORITY_LINE_ERROR_ONLY_ERR"
BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL = (
    "BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR"
)


@dataclass(frozen=True)
class MetaLearnerBaselineColumns:
    avg_all_6_error: str = BASE_AVG_ALL_6_ERR_COL
    majority_total_only_error: str = BASE_MAJORITY_TOTAL_ONLY_ERR_COL
    majority_line_error_only_error: str = BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL
    majority_all_6_tie_line_error_full_dataset_error: str = (
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL
    )


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str],
    line_error_model_cols: list[str],
    line_col: str,
    reference_tiebreaker_col: str,
) -> None:
    required = (
        list(total_model_cols)
        + list(line_error_model_cols)
        + [line_col, reference_tiebreaker_col]
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            "The meta-learner dataframe is missing required baseline columns: "
            f"{missing}"
        )


def build_total_points_error_space_predictions(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str] | None = None,
    line_col: str | None = None,
) -> pd.DataFrame:
    """Convert total-points model outputs into line-error space."""
    resolved_total_model_cols = total_model_cols or list(DEFAULT_TOTAL_POINTS_MODEL_COLS)
    resolved_line_col = line_col or total_line_col()

    missing = [
        column
        for column in [resolved_line_col, *resolved_total_model_cols]
        if column not in df.columns
    ]
    if missing:
        raise KeyError(
            "The meta-learner dataframe is missing required total-points columns: "
            f"{missing}"
        )

    line_values = pd.to_numeric(df[resolved_line_col], errors="coerce")
    converted: dict[str, pd.Series] = {}
    for column in resolved_total_model_cols:
        converted[f"{column}__ERR"] = pd.to_numeric(df[column], errors="coerce") - line_values
    return pd.DataFrame(converted, index=df.index)


def build_all_6_error_space_predictions(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str] | None = None,
    line_error_model_cols: list[str] | None = None,
    line_col: str | None = None,
) -> pd.DataFrame:
    """Return the six model predictions in a common line-error space."""
    resolved_total_model_cols = total_model_cols or list(DEFAULT_TOTAL_POINTS_MODEL_COLS)
    resolved_line_error_model_cols = line_error_model_cols or list(
        DEFAULT_LINE_ERROR_MODEL_COLS
    )
    resolved_line_col = line_col or total_line_col()

    total_error_df = build_total_points_error_space_predictions(
        df,
        total_model_cols=resolved_total_model_cols,
        line_col=resolved_line_col,
    )
    line_error_df = pd.DataFrame(
        {
            f"{column}__ERR": pd.to_numeric(df[column], errors="coerce")
            for column in resolved_line_error_model_cols
        },
        index=df.index,
    )
    return pd.concat([total_error_df, line_error_df], axis=1)


def build_line_error_only_error_space_predictions(
    df: pd.DataFrame,
    *,
    line_error_model_cols: list[str] | None = None,
) -> pd.DataFrame:
    resolved_line_error_model_cols = line_error_model_cols or list(
        DEFAULT_LINE_ERROR_MODEL_COLS
    )
    missing = [
        column for column in resolved_line_error_model_cols if column not in df.columns
    ]
    if missing:
        raise KeyError(
            "The meta-learner dataframe is missing required line-error columns: "
            f"{missing}"
        )

    return pd.DataFrame(
        {
            f"{column}__ERR": pd.to_numeric(df[column], errors="coerce")
            for column in resolved_line_error_model_cols
        },
        index=df.index,
    )


def _build_majority_vote_error_series(
    error_df: pd.DataFrame,
    *,
    reference_error: np.ndarray,
    output_name: str,
) -> pd.Series:
    signs = np.sign(error_df.to_numpy(dtype=float))
    majority = np.sign(signs.sum(axis=1))

    tie_mask = majority == 0
    if np.any(tie_mask):
        majority[tie_mask] = np.sign(reference_error[tie_mask])

    return pd.Series(majority, index=error_df.index, name=output_name)


def build_base_avg_all_6_error(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str] | None = None,
    line_error_model_cols: list[str] | None = None,
    line_col: str | None = None,
    output_name: str = BASE_AVG_ALL_6_ERR_COL,
) -> pd.Series:
    """Average the six model predictions after converting all of them to error space."""
    error_df = build_all_6_error_space_predictions(
        df,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    return pd.Series(
        error_df.mean(axis=1).to_numpy(dtype=float),
        index=df.index,
        name=output_name,
    )


def build_base_majority_total_only_error(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str] | None = None,
    line_col: str | None = None,
    reference_tiebreaker_col: str = DEFAULT_TOTAL_POINTS_MODEL_COLS[0],
    output_name: str = BASE_MAJORITY_TOTAL_ONLY_ERR_COL,
) -> pd.Series:
    resolved_total_model_cols = total_model_cols or list(DEFAULT_TOTAL_POINTS_MODEL_COLS)
    resolved_line_col = line_col or total_line_col()

    missing = [
        column
        for column in [resolved_line_col, *resolved_total_model_cols, reference_tiebreaker_col]
        if column not in df.columns
    ]
    if missing:
        raise KeyError(
            "The meta-learner dataframe is missing required total-points baseline columns: "
            f"{missing}"
        )

    error_df = build_total_points_error_space_predictions(
        df,
        total_model_cols=resolved_total_model_cols,
        line_col=resolved_line_col,
    )
    reference_error = (
        pd.to_numeric(df[reference_tiebreaker_col], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(df[resolved_line_col], errors="coerce").to_numpy(dtype=float)
    )
    return _build_majority_vote_error_series(
        error_df,
        reference_error=reference_error,
        output_name=output_name,
    )


def build_base_majority_line_error_only_error(
    df: pd.DataFrame,
    *,
    line_error_model_cols: list[str] | None = None,
    reference_tiebreaker_col: str = DEFAULT_REFERENCE_TIEBREAKER_COL,
    output_name: str = BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL,
) -> pd.Series:
    resolved_line_error_model_cols = line_error_model_cols or list(
        DEFAULT_LINE_ERROR_MODEL_COLS
    )

    missing = [
        column
        for column in [*resolved_line_error_model_cols, reference_tiebreaker_col]
        if column not in df.columns
    ]
    if missing:
        raise KeyError(
            "The meta-learner dataframe is missing required line-error baseline columns: "
            f"{missing}"
        )

    error_df = build_line_error_only_error_space_predictions(
        df,
        line_error_model_cols=resolved_line_error_model_cols,
    )
    reference_error = pd.to_numeric(
        df[reference_tiebreaker_col], errors="coerce"
    ).to_numpy(dtype=float)
    return _build_majority_vote_error_series(
        error_df,
        reference_error=reference_error,
        output_name=output_name,
    )


def build_base_majority_all_6_tie_line_error_full_dataset_error(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str] | None = None,
    line_error_model_cols: list[str] | None = None,
    line_col: str | None = None,
    reference_tiebreaker_col: str = DEFAULT_REFERENCE_TIEBREAKER_COL,
    output_name: str = BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
) -> pd.Series:
    """
    Majority vote across the six models in error space.

    Exact 3-vs-3 ties are resolved using the sign of the full-dataset line-error
    model prediction, matching the notebook baseline.
    """
    resolved_total_model_cols = total_model_cols or list(DEFAULT_TOTAL_POINTS_MODEL_COLS)
    resolved_line_error_model_cols = line_error_model_cols or list(
        DEFAULT_LINE_ERROR_MODEL_COLS
    )
    resolved_line_col = line_col or total_line_col()

    _validate_required_columns(
        df,
        total_model_cols=resolved_total_model_cols,
        line_error_model_cols=resolved_line_error_model_cols,
        line_col=resolved_line_col,
        reference_tiebreaker_col=reference_tiebreaker_col,
    )

    error_df = build_all_6_error_space_predictions(
        df,
        total_model_cols=resolved_total_model_cols,
        line_error_model_cols=resolved_line_error_model_cols,
        line_col=resolved_line_col,
    )
    reference_error = pd.to_numeric(
        df[reference_tiebreaker_col], errors="coerce"
    ).to_numpy(dtype=float)
    return _build_majority_vote_error_series(
        error_df,
        reference_error=reference_error,
        output_name=output_name,
    )


def add_default_meta_learner_baselines(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str] | None = None,
    line_error_model_cols: list[str] | None = None,
    line_col: str | None = None,
    reference_tiebreaker_col: str = DEFAULT_REFERENCE_TIEBREAKER_COL,
) -> pd.DataFrame:
    """Return a copy of *df* with the two default reusable baseline columns added."""
    output = df.copy()
    output[BASE_AVG_ALL_6_ERR_COL] = build_base_avg_all_6_error(
        output,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    output[BASE_MAJORITY_TOTAL_ONLY_ERR_COL] = build_base_majority_total_only_error(
        output,
        total_model_cols=total_model_cols,
        line_col=line_col,
    )
    output[BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL] = (
        build_base_majority_line_error_only_error(
            output,
            line_error_model_cols=line_error_model_cols,
            reference_tiebreaker_col=reference_tiebreaker_col,
        )
    )
    output[
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL
    ] = build_base_majority_all_6_tie_line_error_full_dataset_error(
        output,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
        reference_tiebreaker_col=reference_tiebreaker_col,
    )
    return output
