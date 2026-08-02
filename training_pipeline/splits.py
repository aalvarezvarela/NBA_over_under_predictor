"""Thin dispatch over the existing CV/holdout builders in nba_ou.modeling.modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nba_ou.modeling.modeling import (
    assert_valid_time_splits,
    make_test_anchored_walk_forward_splits,
    make_walk_forward_last_n_seasons_splits,
    split_latest_dates_holdout,
)

from training_pipeline.config import CVStrategy, ExperimentConfig
from training_pipeline.data import training_eligible_mask


def split_latest_days_holdout(
    df: pd.DataFrame, *, date_col: str, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fixed calendar window from the end of the data.

    The cut is on the date, not on a row count, so every game on the boundary
    day lands on the same side of the split -- a count-based cut can slice a
    game-day in half, putting some of a day's games in training and the rest in
    test, which is exactly the leak the daily walk-forward exists to avoid.

    Counted back from the last game present, so the window is defined by the
    data rather than by today's date; re-running an old config gives the same
    split.
    """
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.max() - pd.Timedelta(days=test_days)

    df_dev = df.loc[dates <= cutoff].copy().reset_index(drop=True)
    df_test = df.loc[dates > cutoff].copy().reset_index(drop=True)

    if df_test.empty:
        raise ValueError(
            f"holdout.test_days={test_days} selected no games. The data ends "
            f"{dates.max().date()}."
        )
    if df_dev.empty:
        raise ValueError(
            f"holdout.test_days={test_days} consumed the entire dataset "
            f"({dates.min().date()} to {dates.max().date()}); nothing left to "
            "train on."
        )
    return df_dev, df_test


def build_holdout_split(
    df_full: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.holdout.test_days is not None:
        return split_latest_days_holdout(
            df_full,
            date_col=config.data.date_col,
            test_days=config.holdout.test_days,
        )
    return split_latest_dates_holdout(
        df=df_full,
        date_col=config.data.date_col,
        test_size=config.holdout.test_size,
        test_games=config.holdout.test_games,
    )


def build_walk_forward_splits(
    df_dev: pd.DataFrame, config: ExperimentConfig
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    wf = config.walk_forward

    if wf.strategy == CVStrategy.TEST_ANCHORED:
        splits, fold_info = make_test_anchored_walk_forward_splits(
            df=df_dev,
            date_col=config.data.date_col,
            season_col=config.data.season_col,
            test_games=wf.test_games,
            step_games_between_tests=wf.step_games_between_tests,
            train_games=wf.train_games,
            min_train_games=wf.min_train_games,
            exclude_test_months=wf.exclude_test_months,
            require_same_season_test=wf.require_same_season_test,
            max_folds=wf.max_folds,
            fold_selection=wf.fold_selection,
            verbose=wf.verbose,
        )
    else:  # LAST_N_SEASONS
        splits, fold_info = make_walk_forward_last_n_seasons_splits(
            df=df_dev,
            date_col=config.data.date_col,
            season_col=config.data.season_col,
            train_seasons=wf.train_seasons,
            test_games=wf.test_games,
            step_games=wf.step_games_between_tests,
            min_train_games=wf.min_train_games,
            max_folds=wf.max_folds,
            fold_selection=wf.fold_selection,
            verbose=wf.verbose,
        )

    # Applied to the TRAIN half of each fold only. This single hook covers both
    # the Optuna objective and cv_betting, which share these splits -- so the
    # hyperparameters are selected under exactly the training regime the final
    # model will use, while every fold is still SCORED on all its games.
    splits = apply_training_filter(df_dev, splits, config)

    validate_splits(df_dev, splits, date_col=config.data.date_col)
    return splits, fold_info


def apply_training_filter(
    df_dev: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop training-ineligible rows from each fold's TRAIN indices.

    Validation indices are returned untouched, which is the whole point: the
    filter changes what the model learns from, never what it is judged on.
    """
    mask = training_eligible_mask(df_dev, config)
    if mask.all():
        return splits

    filtered = []
    for train_idx, valid_idx in splits:
        kept = train_idx[mask[train_idx]]
        if len(kept) == 0:
            raise ValueError(
                "A CV fold has no training rows left after filtering overtime "
                "games. Widen walk_forward.train_games or turn "
                "data.exclude_overtime_from_training off."
            )
        filtered.append((kept, valid_idx))
    return filtered


def validate_splits(
    df_dev: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    date_col: str,
) -> None:
    assert_valid_time_splits(df=df_dev, splits=splits, date_col=date_col)
