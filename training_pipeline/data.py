"""Data loading, cleaning, and feature/target preparation for training_pipeline.

Wraps nba_ou.data_processing.missing_data.clean_df_for_training.clean_dataframe_for_training
and reuses nba_ou.modeling.meta_learner_training_data._ensure_line_error_column as the
single canonical LINE_ERROR derivation, rather than re-implementing it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.config.odds_columns import resolve_main_total_line_col, total_line_col
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)

# Reused, not duplicated: this is documented as the one canonical implementation
# of LINE_ERROR = TOTAL_POINTS - TOTAL_LINE_<book> in the repo. The leading
# underscore is a naming convention, not an enforced boundary; duplicating the
# formula here would risk the two definitions drifting apart over time.
from nba_ou.modeling.meta_learner_training_data import (
    _ensure_line_error_column as ensure_line_error_column,
)

from training_pipeline.config import BaselineConfig, CleaningConfig, ExperimentConfig

# NOTE: "odds_total_line_books_median" (the engineered cross-book median total
# line, per src/nba_ou/data_processing/merged_home_away_data/odds_feature_engeneer.py)
# was the original default candidate for the baseline line column. It is
# deliberately NOT used as a silent default: verified empirically against a
# real archived training CSV (data/train_data/all_odds_training_data_until_20260318.csv)
# that this column's values (range roughly -0.5 to 17) do not correlate with
# TOTAL_POINTS or TOTAL_LINE_bet365 (correlation ~0.02) for that snapshot --
# whatever generated it did not produce a points-scale median. Since historical
# CSV snapshots cannot be assumed trustworthy for this column, it is only used
# as the baseline when a caller explicitly opts in via BaselineConfig.line_col.
BOOKMAKER_MEDIAN_LINE_COL = "odds_total_line_books_median"


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
            f"'{BOOKMAKER_MEDIAN_LINE_COL}' column and no TOTAL_LINE_<book> "
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
    if config.target_family.value == "line_error":
        keep.add("LINE_ERROR")
    for price_col in (config.betting.over_price_col, config.betting.under_price_col):
        if price_col:
            keep.add(price_col)
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

    if config.target_family.value == "line_error":
        df = ensure_line_error_column(df)
        target_col = "LINE_ERROR"
        # _ensure_line_error_column subtracts the configured main book's line,
        # so that is the line bets must be settled into for this target.
        target_line_col = total_line_col()
    else:
        target_col = "TOTAL_POINTS"
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        target_line_col = config.line_col

    baseline_line_col = resolve_baseline_line_col(df, config.baseline)

    force_keep = _required_keep_columns(config, baseline_line_col, target_line_col)
    df = clean_for_training(df, config.cleaning, force_keep_columns=force_keep)

    if target_line_col not in df.columns:
        raise KeyError(
            f"Target line column {target_line_col!r} is missing after cleaning; "
            "bets cannot be settled without it."
        )

    dropna_subset = sorted({target_col, target_line_col, baseline_line_col})
    df = df.dropna(subset=dropna_subset).copy()

    df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])
    df = df.sort_values(config.data.date_col).reset_index(drop=True)

    X, y = build_feature_matrix(df, target_col=target_col, exclude_cols=config.exclude_cols)

    return PreparedDataset(
        df_full=df,
        X=X,
        y=y,
        baseline_line_col=baseline_line_col,
        target_line_col=target_line_col,
        feature_names=list(X.columns),
        dataset_checksum=dataset_checksum,
    )
