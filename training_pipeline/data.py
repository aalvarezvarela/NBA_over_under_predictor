"""Data loading, cleaning, and feature/target preparation for training_pipeline.

Wraps nba_ou.data_processing.missing_data.clean_df_for_training.clean_dataframe_for_training
and reuses nba_ou.modeling.meta_learner_training_data._ensure_line_error_column as the
single canonical LINE_ERROR derivation, rather than re-implementing it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.config.odds_columns import resolve_main_total_line_col, total_line_col
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)

# Reused, not duplicated: this is documented as the one canonical implementation
# of LINE_ERROR = TOTAL_POINTS - ODDS_TOTAL_LINE_<book> in the repo. The leading
# underscore is a naming convention, not an enforced boundary; duplicating the
# formula here would risk the two definitions drifting apart over time.
from nba_ou.modeling.meta_learner_training_data import (
    _ensure_line_error_column as ensure_line_error_column,
)

from training_pipeline.config import (
    LEAKING_TARGET_COLUMNS,
    OVER_LABEL_COL,
    BaselineConfig,
    CleaningConfig,
    ExperimentConfig,
    PredictionStrategy,
)
from training_pipeline.diagnostics import (
    PlantedSignalResult,
    build_planted_signal,
    measure_planted_signal,
)

# NOTE: "ODDS_total_line_books_median" (the engineered cross-book median total
# line, per src/nba_ou/data_processing/merged_home_away_data/odds_feature_engeneer.py)
# was the original default candidate for the baseline line column. It is
# deliberately NOT used as a silent default: verified empirically against a
# real archived training CSV (data/train_data/all_odds_training_data_until_20260318.csv)
# that this column's values (range roughly -0.5 to 17) do not correlate with
# TOTAL_POINTS or ODDS_TOTAL_LINE_bet365 (correlation ~0.02) for that snapshot --
# whatever generated it did not produce a points-scale median. Since historical
# CSV snapshots cannot be assumed trustworthy for this column, it is only used
# as the baseline when a caller explicitly opts in via BaselineConfig.line_col.
BOOKMAKER_MEDIAN_LINE_COL = "ODDS_total_line_books_median"


def compute_file_checksum(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streamed sha256 of a file's contents, as ``sha256:<first 16 hex>``.

    Content-based rather than path-based: training CSVs get regenerated in
    place under the same filename, and a path alone cannot tell you whether
    the bytes behind it changed. Streaming keeps memory flat on the ~400MB
    snapshots; the read costs well under a second.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


def verify_dataset_checksum(
    path: str | Path, *, expected_checksum: str | None
) -> str:
    """Compute the dataset checksum, asserting it matches when one is pinned."""
    actual = compute_file_checksum(path)
    if expected_checksum and actual != expected_checksum:
        raise ValueError(
            f"Dataset checksum mismatch for {path}: expected {expected_checksum}, "
            f"got {actual}. The file's contents changed since this experiment "
            "was defined. Update data.expected_checksum if that was intentional."
        )
    return actual


def load_raw_training_csv(csv_path: str | Path, *, date_col: str = "GAME_DATE") -> pd.DataFrame:
    """Load a training CSV the same way the example notebooks do: ID-like columns
    forced to str (avoids mixed-type surprises from pandas' dtype inference on
    large sparse columns), GAME_DATE parsed to a plain date string then back to
    datetime for consistent downstream handling.
    """
    csv_path = Path(csv_path)
    header = pd.read_csv(csv_path, nrows=0)
    dtype_dict = {col: str for col in header.columns if "ID" in col.upper()}

    df = pd.read_csv(csv_path, dtype=dtype_dict)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def apply_season_year_floor(
    df: pd.DataFrame, *, season_col: str = "SEASON_YEAR", floor: int | None
) -> pd.DataFrame:
    if floor is None:
        return df
    return df[df[season_col] >= floor].copy()


def resolve_season_type(df: pd.DataFrame, *, game_id_col: str = "GAME_ID") -> pd.Series:
    """Season type derived from the GAME_ID prefix.

    This is deliberately NOT read from the ``SEASON_TYPE`` text column. In the
    training data that column labels Play-In Tournament games as "Playoffs"
    (verified: all 31 rows with GAME_ID prefix 005 carry SEASON_TYPE
    "Playoffs"), so filtering on it would silently discard the play-in games we
    want to keep. The GAME_ID prefix maps cleanly through
    nba_ou.config.constants.SEASON_TYPE_MAP: 002 regular season, 004 playoffs,
    005 play-in, 006 in-season final.

    Unmappable prefixes yield NaN.
    """
    if game_id_col not in df.columns:
        raise KeyError(
            f"Cannot determine season type: column {game_id_col!r} is missing. "
            "Set data.exclude_playoffs=False to skip season-type filtering, or "
            "point data.game_id_col at the right column."
        )
    return df[game_id_col].astype(str).str.strip().str[:3].map(SEASON_TYPE_MAP)


def filter_allowed_season_types(
    df: pd.DataFrame,
    *,
    allowed_season_types: tuple[str, ...],
    game_id_col: str = "GAME_ID",
) -> pd.DataFrame:
    """Keep only games whose season type is in ``allowed_season_types``.

    Rows whose GAME_ID prefix is unknown are dropped, since an unrecognised
    competition type is not something the model should silently train on.
    """
    season_types = resolve_season_type(df, game_id_col=game_id_col)
    keep = season_types.isin(allowed_season_types)
    return df.loc[keep].copy()


def resolve_baseline_line_col(df: pd.DataFrame, baseline: BaselineConfig) -> str:
    """Resolve which column stands in for "trust the bookmaker's line".

    Priority: explicit override (this is how a caller opts into
    BOOKMAKER_MEDIAN_LINE_COL or any other cross-book aggregate) -> the
    resolved main total-line column (the same single-book line already used
    everywhere else in the pipeline for cleaning and OU-accuracy scoring, and
    the only choice verified trustworthy across CSV snapshots; works for both
    the "wide" per-book and "reduced" single-column CSV schemas).
    """
    if baseline.line_col:
        if baseline.line_col not in df.columns:
            raise KeyError(
                f"baseline.line_col={baseline.line_col!r} not found in the loaded data."
            )
        return baseline.line_col

    resolved = resolve_main_total_line_col(df, book=baseline.book)
    if resolved is None:
        raise KeyError(
            "Could not resolve a baseline line column: no "
            f"'{BOOKMAKER_MEDIAN_LINE_COL}' column and no ODDS_TOTAL_LINE_<book> "
            "column found. Set baseline.line_col explicitly."
        )
    return resolved


def _required_keep_columns(
    config: ExperimentConfig, baseline_line_col: str, target_line_col: str
) -> list[str]:
    keep = {
        config.data.date_col,
        config.data.season_col,
        "TOTAL_POINTS",
        baseline_line_col,
        target_line_col,
    }
    if config.line_col:
        keep.add(config.line_col)
    if config.strategy == PredictionStrategy.LINE_ERROR_REGRESSOR:
        keep.add("LINE_ERROR")
    for price_col in (config.betting.over_price_col, config.betting.under_price_col):
        if price_col:
            keep.add(price_col)
    # Alternative lines to re-score against. These are especially exposed to
    # correlation pruning -- an opening line is near-perfectly correlated with
    # the closing line, which is exactly why it would be dropped, and exactly
    # why the comparison is interesting.
    keep.update(config.betting.comparison_line_cols)
    # Needed as a ROW attribute to filter training on, so it must survive
    # cleaning even though it is never a feature.
    if config.data.exclude_overtime_from_training:
        keep.add(config.data.overtime_col)
    return sorted(keep)


def clean_for_training(
    df: pd.DataFrame,
    cleaning: CleaningConfig,
    *,
    force_keep_columns: list[str],
) -> pd.DataFrame:
    """Wrap clean_dataframe_for_training, guaranteeing force_keep_columns survive.

    clean_dataframe_for_training's keep_columns argument only protects columns
    from the string/ID/NAME/high-NaN/constant-column removal steps -- it does
    NOT protect against the duplicate-column, correlation-pruning, or
    absolute-value-match steps (verified by reading
    nba_ou/data_processing/missing_data/clean_df_for_training.py directly:
    those steps never check keep_columns_set). Without this safeguard, the
    baseline line column could be silently dropped for being highly
    correlated with a per-book line column, breaking baseline comparability
    without any error. So: snapshot required columns before cleaning, and
    reattach any that didn't survive, aligned to the cleaned frame's index.
    """
    keep_columns = sorted(set(cleaning.keep_columns or []) | set(force_keep_columns))
    present_before = [c for c in force_keep_columns if c in df.columns]
    snapshot = df[present_before].copy()

    cleaned = clean_dataframe_for_training(
        df,
        nan_threshold=cleaning.nan_threshold,
        corr_threshold=cleaning.corr_threshold,
        max_na_per_row=cleaning.max_na_per_row,
        create_missing_flags=cleaning.create_missing_flags,
        keep_columns=keep_columns,
        exclude_cols_containing=cleaning.exclude_cols_containing,
        keep_all_cols=cleaning.keep_all_cols,
        verbose=cleaning.verbose,
        strict_mode=cleaning.strict_mode,
        strict_mode_exclude_cols=cleaning.strict_mode_exclude_cols,
    )

    missing_after = [c for c in present_before if c not in cleaned.columns]
    if missing_after:
        reattach = snapshot.loc[cleaned.index, missing_after]
        cleaned = pd.concat([cleaned, reattach], axis=1)

    return cleaned


def training_eligible_mask(
    df: pd.DataFrame, config: ExperimentConfig
) -> np.ndarray:
    """Boolean per row: may this game be used to FIT a model?

    Evaluation ignores this entirely. Validation folds, the holdout and the
    walk-forward's prediction days always score every game, because ~5.2% of
    real games go to overtime and you are paid or not paid on those -- dropping
    them from scoring would measure a world that does not exist.

    Returns all-True when the filter is off, so callers need no branch.
    """
    if not config.data.exclude_overtime_from_training:
        return np.ones(len(df), dtype=bool)

    column = config.data.overtime_col
    if column not in df.columns:
        raise KeyError(
            f"data.exclude_overtime_from_training is on but {column!r} is not in "
            "the training data. It is dropped by training-data builds predating "
            "this option -- regenerate the CSV with "
            "scripts/create_train_data/create_train_data.py."
        )
    flag = pd.to_numeric(df[column], errors="coerce")
    # NaN means "unknown", which for a scheduled game means "not yet played".
    # Treat only a definite 1 as overtime so unknowns stay trainable.
    return (flag != 1).to_numpy(dtype=bool)


def assert_no_leaking_features(X: pd.DataFrame) -> None:
    """Fail loudly if an outcome-derived column reached the feature matrix.

    ``exclude_cols`` already lists these, but it is a config field a caller can
    overwrite, and the columns only appear in some CSV snapshots -- exactly the
    combination that produces a silent, spectacular-looking result rather than
    an error. This mirrors the ``_BEFORE`` leakage guard that
    ``select_training_columns`` already enforces upstream.

    Exact matches only: ``DIFF_FROM_LINE_*_BEFORE_*`` are legitimate pre-game
    rollups, and a substring check would discard hundreds of real features.
    """
    leaked = sorted(set(X.columns) & set(LEAKING_TARGET_COLUMNS))
    if leaked:
        raise ValueError(
            f"Outcome-derived column(s) {leaked} reached the feature matrix. "
            "These are functions of the final score, so a model trained on them "
            "would look excellent and predict nothing. Add them to "
            "config.exclude_cols. (Engineered *_BEFORE_* rollups are unaffected.)"
        )


def add_over_under_label(
    df: pd.DataFrame, *, line_col: str
) -> tuple[pd.DataFrame, int]:
    """Attach the binary OVER label, dropping pushes.

    ``1`` when the total beat the line, ``0`` when it fell short. Games landing
    exactly on the line are removed: they have no OVER/UNDER answer, so a label
    would have to be invented, and inventing one teaches the model a fiction on
    the very rows where the market was most precisely right.

    Dropping them costs almost nothing here -- measured on the 2.0 training
    data, pushes are 1.175% of games, because 53.3% of lines end in .5 and
    cannot push at all, and whole-number lines only push 2.52% of the time.
    That also settles the question of whether a three-outcome model is worth
    building: at this rate, it is not.

    Pushes are dropped from TRAINING only. Betting evaluation keeps them, where
    they are scored the way a sportsbook settles them -- stake returned,
    excluded from the win rate -- so profitability stays honest.

    Returns the frame and the number of pushes removed.
    """
    total = pd.to_numeric(df["TOTAL_POINTS"], errors="coerce")
    line = pd.to_numeric(df[line_col], errors="coerce")
    margin = total - line

    is_push = margin == 0
    n_pushes = int(is_push.sum())

    labelled = df.loc[~is_push].copy()
    labelled[OVER_LABEL_COL] = (
        (pd.to_numeric(labelled["TOTAL_POINTS"], errors="coerce")
         - pd.to_numeric(labelled[line_col], errors="coerce")) > 0
    ).astype(int)

    if labelled.empty:
        raise ValueError(
            f"No rows left after dropping pushes against {line_col!r}."
        )
    return labelled, n_pushes


def build_feature_matrix(
    df: pd.DataFrame, *, target_col: str, exclude_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=exclude_cols, errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")
    return X, y


@dataclass(frozen=True)
class PreparedDataset:
    df_full: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    baseline_line_col: str
    #: The line the target is defined against and that bets are settled into.
    #: For TOTAL_POINTS this is the configured scoring line; for LINE_ERROR it
    #: is whatever line _ensure_line_error_column subtracted, i.e. the main
    #: book. Kept separate from ``baseline_line_col``, which may deliberately
    #: point at an alternative consensus line for the MAE baseline.
    target_line_col: str
    feature_names: list[str]
    #: sha256 of the source CSV actually read, recorded in run metadata so a
    #: saved run can be tied back to the exact bytes it trained on.
    dataset_checksum: str | None = None
    #: Games dropped because the total landed exactly on the line, which has no
    #: OVER/UNDER answer to learn. Classifier only; 0 for the regressors, which
    #: keep those rows (a push is a perfectly good regression target).
    n_pushes_excluded: int = 0
    #: What the planted diagnostic feature actually ended up carrying. None on
    #: every normal run -- its presence is the marker that this dataset is
    #: deliberately corrupted and cannot be read as evidence about the market.
    planted_signal: PlantedSignalResult | None = None


def prepare_dataset(config: ExperimentConfig) -> PreparedDataset:
    dataset_checksum = verify_dataset_checksum(
        config.data.csv_path, expected_checksum=config.data.expected_checksum
    )
    df = load_raw_training_csv(config.data.csv_path, date_col=config.data.date_col)
    df = apply_season_year_floor(
        df, season_col=config.data.season_col, floor=config.data.season_year_floor
    )

    # Must run before cleaning: advanced_column_cleaning drops both GAME_ID
    # (name contains "_ID") and SEASON_TYPE (a pure-string column), so the
    # information needed to identify competition type is gone afterwards.
    if config.data.exclude_playoffs:
        df = filter_allowed_season_types(
            df,
            allowed_season_types=config.data.allowed_season_types,
            game_id_col=config.data.game_id_col,
        )
        if df.empty:
            raise ValueError(
                "No rows left after season-type filtering. Check "
                f"data.allowed_season_types={config.data.allowed_season_types}."
            )

    if config.strategy == PredictionStrategy.LINE_ERROR_REGRESSOR:
        df = ensure_line_error_column(df)
        # _ensure_line_error_column subtracts the configured main book's line,
        # so that is the line bets must be settled into for this target.
        target_line_col = total_line_col()
    else:
        # Both the total-points regressor and the classifier settle into the
        # explicitly configured line. For the classifier that line is part of
        # the label's definition, not merely a scoring choice.
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        target_line_col = config.line_col
    target_col = config.target_col

    baseline_line_col = resolve_baseline_line_col(df, config.baseline)

    # Planted BEFORE cleaning, and deliberately NOT added to force_keep: the
    # point of the diagnostic is that the synthetic feature travels the same
    # path as a real one -- the NaN budget, the correlation prune, the constant
    # and duplicate-column steps, then build_feature_matrix. Force-keeping it
    # would exempt it from exactly what is being tested. It is checked for
    # survival below instead, so a drop is an error rather than a quiet null
    # result.
    planted = config.diagnostics.planted_signal
    if planted.enabled:
        if target_col not in df.columns:
            raise KeyError(
                f"Cannot plant a signal: target column {target_col!r} is not in "
                "the frame yet. The planted feature must be derived after the "
                "target exists and before cleaning."
            )
        df = df.copy()
        df[planted.column] = build_planted_signal(df[target_col], config=planted)

    force_keep = _required_keep_columns(config, baseline_line_col, target_line_col)
    df = clean_for_training(df, config.cleaning, force_keep_columns=force_keep)

    if planted.enabled and planted.column not in df.columns:
        raise ValueError(
            f"The planted feature {planted.column!r} did not survive cleaning, so "
            "the diagnostic would measure nothing while appearing to run. Check "
            "cleaning.exclude_cols_containing and cleaning.corr_threshold."
        )

    if target_line_col not in df.columns:
        raise KeyError(
            f"Target line column {target_line_col!r} is missing after cleaning; "
            "bets cannot be settled without it."
        )

    # The label depends on TOTAL_POINTS and the line, so it must be derived
    # after cleaning has guaranteed both survive, and before the target column
    # is required to exist by the dropna below.
    n_pushes_excluded = 0
    if config.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
        df, n_pushes_excluded = add_over_under_label(
            df, line_col=target_line_col
        )

    dropna_subset = sorted({target_col, target_line_col, baseline_line_col})
    df = df.dropna(subset=dropna_subset).copy()

    df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])
    df = df.sort_values(config.data.date_col).reset_index(drop=True)

    X, y = build_feature_matrix(df, target_col=target_col, exclude_cols=config.exclude_cols)
    assert_no_leaking_features(X)

    planted_result = None
    if planted.enabled:
        if planted.column not in X.columns:
            raise ValueError(
                f"The planted feature {planted.column!r} was cleaned through but "
                "never reached the feature matrix, so no model would ever see "
                "it. Check config.exclude_cols."
            )
        # Measured on the frame the model actually gets, not on the frame the
        # feature was generated against: rows are dropped in between, so the
        # realised correlation is the honest number.
        planted_result = measure_planted_signal(
            df, target_col=target_col, config=planted
        )

    return PreparedDataset(
        df_full=df,
        X=X,
        y=y,
        baseline_line_col=baseline_line_col,
        target_line_col=target_line_col,
        feature_names=list(X.columns),
        dataset_checksum=dataset_checksum,
        n_pushes_excluded=n_pushes_excluded,
        planted_signal=planted_result,
    )
