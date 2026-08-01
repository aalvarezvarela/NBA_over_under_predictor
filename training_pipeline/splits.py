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


def build_holdout_split(
    df_full: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    validate_splits(df_dev, splits, date_col=config.data.date_col)
    return splits, fold_info


def validate_splits(
    df_dev: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    date_col: str,
) -> None:
    assert_valid_time_splits(df=df_dev, splits=splits, date_col=date_col)
