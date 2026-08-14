#!/usr/bin/env python3
"""Create the intermediate-line training dataset (one row per game AND snapshot).

Separate from ``create_train_data.py`` in every respect: separate entry point,
separate output file, no shared state. Building one dataset cannot affect the
other.

    poetry run python scripts/create_train_data/create_intermediate_line_train_data.py

The printed sha256 goes straight into a campaign config's
``data.expected_checksum`` so a regenerated CSV cannot pass silently.

**Read the row-count warning it prints.** Every window in
``experiments/_base.yaml`` named ``*_games`` is counted in ROWS, not games
(``train_pool.tail(train_games)``), so on a dataset with N snapshots per game
the inherited defaults cover 1/N of the calendar history they were reasoned
for -- with no error raised. The script prints the rescaled values to use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.create_training_data.create_intermediate_line_df import (
    create_intermediate_line_df,
)
from nba_ou.data_processing.line_history.movement_features import DEFAULT_WINDOWS
from nba_ou.data_processing.line_history.snapshots import DEFAULT_SNAPSHOT_GRID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "train_data"

#: Settings in experiments/_base.yaml that are counted in rows despite their
#: names. holdout.test_days is deliberately absent: it is calendar-based and is
#: the one window that needs no rescaling.
ROW_COUNTED_SETTINGS: dict[str, int] = {
    "walk_forward.train_games": 2500,
    "walk_forward.min_train_games": 1250,
    "walk_forward.test_games": 50,
    "walk_forward.step_games_between_tests": 60,
    "backtest.test_games": 300,
}


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(",") if part.strip())


def print_config_guidance(n_snapshots: int) -> None:
    print()
    print("=" * 68)
    print(f"Row multiplier: {n_snapshots} snapshots per game.")
    print("Rescale these in your campaign config -- they count ROWS, not games:")
    for setting, base_value in ROW_COUNTED_SETTINGS.items():
        print(f"  {setting:<40} {base_value:>6} -> {base_value * n_snapshots:>6}")
    print("  holdout.test_days                              60 ->     60  (calendar)")
    print()
    print("Then run the pre-flight and read the ACTUAL per-fold training size:")
    print("  poetry run python scripts/preflight_campaign.py experiments/<campaign>")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated season years. Default: everything in the store.",
    )
    parser.add_argument("--recent-limit", type=str, default=None)
    parser.add_argument(
        "--snapshot-grid",
        type=str,
        default=",".join(str(m) for m in DEFAULT_SNAPSHOT_GRID),
        help="Minutes before tip to sample.",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=",".join(str(m) for m in DEFAULT_WINDOWS),
        help="Trailing look-back windows in minutes.",
    )
    parser.add_argument(
        "--anchor-book",
        type=str,
        default=None,
        help=(
            "Book whose snapshot line defines the target and settles bets. "
            "'consensus' uses the cross-book median (steadier, but not bettable)."
        ),
    )
    parser.add_argument(
        "--include-fanatics",
        action="store_true",
        help=(
            "Keep fanatics_sportsbook. Off by default: it exists only from 2025, "
            "so its mere presence identifies the season."
        ),
    )
    args = parser.parse_args()

    grid = _parse_int_tuple(args.snapshot_grid)
    windows = _parse_int_tuple(args.windows)
    seasons = (
        [int(s) for s in args.seasons.split(",")] if args.seasons else None
    )

    df, scoring = create_intermediate_line_df(
        recent_limit_to_include=args.recent_limit,
        season_years=seasons,
        snapshot_grid=grid,
        windows=windows,
        anchor_book=args.anchor_book,
        exclude_fanatics=not args.include_fanatics,
        return_scoring=True,
    )

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = pd.to_datetime(df["GAME_DATE"]).max().strftime("%Y%m%d")
        output_path = DEFAULT_OUTPUT_DIR / f"intermediate_line_data_{stamp}.csv"

    df.to_csv(output_path, index=False)
    print(f"\nSaved training data to {output_path}")

    # Closing lines and snapshot weights live in a separate file on purpose:
    # the training pipeline builds X by dropping only configured exclusions, so
    # anything left in the training CSV becomes a feature.
    scoring_path = output_path.with_name(f"{output_path.stem}_scoring.csv")
    scoring.to_csv(scoring_path, index=False)
    print(f"Saved scoring sidecar to {scoring_path} (join on GAME_ID + TIME_TO_MATCH_MIN)")

    from training_pipeline.data import compute_file_checksum

    print(f"expected_checksum: \"{compute_file_checksum(output_path)}\"")

    print_row_retention(df)
    print_config_guidance(df["TIME_TO_MATCH_MIN"].nunique())


def print_row_retention(df: pd.DataFrame) -> None:
    """Per-snapshot NaN load.

    ``cleaning.max_na_per_row`` drops the rows carrying the most NaNs, and the
    long horizons are systematically the ones whose look-back windows do not
    reach far enough. If this column is not roughly flat, cleaning will delete
    precisely the snapshots the dataset exists to compare.
    """
    na_per_row = df.isna().sum(axis=1)
    summary = (
        pd.DataFrame(
            {"TIME_TO_MATCH_MIN": df["TIME_TO_MATCH_MIN"], "na_per_row": na_per_row}
        )
        .groupby("TIME_TO_MATCH_MIN")["na_per_row"]
        .agg(["count", "mean", "max"])
        .round(1)
    )
    print("\nNaNs per row by snapshot (must be roughly flat):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
