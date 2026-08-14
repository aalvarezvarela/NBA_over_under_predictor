#!/usr/bin/env python3
"""Run one training_pipeline experiment and add a per-snapshot betting report.

For the intermediate-line dataset, which holds one row per (game, pre-game
snapshot). ONE model is trained on every snapshot at once -- ``TIME_TO_MATCH_MIN``
is an ordinary feature, so the model learns how the mapping changes with time to
tip and can be used at whatever hour you actually place the bet. The extra
report scores that single model one snapshot at a time, answering "what is my
win rate if I always bet 12 hours out?" separately from "...if I always bet 30
minutes out?".

    poetry run python scripts/run_intermediate_snapshot_experiment.py \
        experiments/intermediate_line_2026_08/pooled.yaml

This is a thin wrapper. It calls the ordinary ``run_experiment`` and then
``training_pipeline.snapshot_scoring`` on the finished result -- no training
code, no evaluation code and no config schema is modified, so every existing
experiment behaves exactly as before.

Why the split scoring is necessary rather than merely interesting: see the
module docstring of ``training_pipeline.snapshot_scoring``. In short,
``evaluate_betting`` counts rows as independent bets, so pooling six snapshots
of one game reports six bets' worth of confidence for one game's worth of
evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from training_pipeline.cli import load_config
from training_pipeline.pipeline import run_experiment
from training_pipeline.snapshot_scoring import (
    SNAPSHOT_COLUMN,
    build_snapshot_report,
    format_snapshot_table,
    save_snapshot_report,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to a YAML experiment config file.")
    parser.add_argument(
        "--snapshot-col",
        default=SNAPSHOT_COLUMN,
        help=f"Column holding minutes before tip (default: {SNAPSHOT_COLUMN}).",
    )
    parser.add_argument(
        "--save-model",
        dest="save_model",
        action="store_true",
        default=None,
        help=(
            "Force a production refit and save the bundle. Default is to let "
            "refit.train_production_model in the config decide, which for the "
            "campaign configs means NO refit."
        ),
    )
    parser.add_argument(
        "--no-save-model",
        dest="save_model",
        action="store_false",
        help="Force-skip the production refit even if the config asks for one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved config; do not train.",
    )
    return parser


def verify_one_row_per_game(config, snapshot_col: str) -> None:
    """Check the invariant the per-snapshot report depends on.

    Within one snapshot there must be exactly one row per game -- that is what
    makes ``n_bets`` a count of independent events and the Wilson interval
    honest. It is checked here, against the raw CSV, rather than inside the
    scoring module: GAME_ID does not survive ``advanced_column_cleaning`` (its
    name contains "_ID"), and it must not, because the default ``exclude_cols``
    would then let a string encoding date-and-sequence into the feature matrix.

    Reading two columns of the CSV settles it directly. Cleaning and filtering
    only ever REMOVE rows, so an invariant that holds on the raw file still
    holds on every subset the pipeline scores.
    """
    columns = pd.read_csv(config.data.csv_path, nrows=0).columns
    if snapshot_col not in columns or "GAME_ID" not in columns:
        print(
            f"  skipped: need GAME_ID and {snapshot_col} in the CSV to check it."
        )
        return

    frame = pd.read_csv(
        config.data.csv_path, usecols=["GAME_ID", snapshot_col], dtype={"GAME_ID": str}
    )
    duplicated = frame.duplicated(subset=["GAME_ID", snapshot_col]).sum()
    if duplicated:
        raise SystemExit(
            f"{duplicated} (GAME_ID, {snapshot_col}) pairs are duplicated in "
            f"{config.data.csv_path}. A snapshot group would then hold the same "
            "game more than once and its confidence interval would be too "
            "narrow. Refusing to run."
        )
    per_snapshot = frame.groupby(snapshot_col)["GAME_ID"].nunique()
    print(
        f"  one row per game per snapshot: OK "
        f"({len(per_snapshot)} snapshots, {per_snapshot.min():,}-"
        f"{per_snapshot.max():,} games each)"
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config(args.config_path)

    if args.dry_run:
        print(config.model_dump_json(indent=2))
        return

    print("Checking the dataset invariant the per-snapshot report relies on...")
    verify_one_row_per_game(config, args.snapshot_col)

    # save_model=None lets refit.train_production_model decide. Passing a bool
    # OVERRIDES the config -- the stock CLI always passes one, so it silently
    # forces a production refit even on a config that turned it off.
    result = run_experiment(config, save_model=args.save_model)

    print(f"\nExperiment: {config.experiment_name}")
    print(f"Target: {config.family.value}")
    evaluation = result.holdout_result or result.walk_forward_result
    if evaluation is not None:
        print(f"Evaluation mode: {config.holdout_evaluation.value}")
        print(f"Test MAE: {evaluation.mae:.4f}")
        primary = evaluation.betting_primary
        if primary.roi is not None:
            print(
                f"POOLED ROI @ edge>{primary.min_edge}: {primary.roi:+.2%} on "
                f"{primary.n_bets} rows -- NOT games; see the table below."
            )

    report = build_snapshot_report(result, snapshot_col=args.snapshot_col)
    if not report:
        print(
            f"\nNo per-snapshot report: column {args.snapshot_col!r} was not found, "
            "or the run produced no prediction frames to group."
        )
        return

    for name, table in report.items():
        title = {
            "cv": "CROSS-VALIDATION FOLDS (pooled validation rows)",
            "holdout": "HELD-OUT TEST PERIOD (daily walk-forward)",
        }.get(name, name.upper())
        print(f"\n{'=' * 78}\n{title}\nby virtual bet time, minutes before tip\n")
        print(format_snapshot_table(table))

    print(
        "\nRows with n_snapshots == 1 are one game per row: n_bets counts "
        "independent\nevents and the interval is honest. The ALL row pools every "
        "snapshot, so it\ncounts one game once per snapshot -- its interval and "
        "significance verdict are\nleft blank rather than reported, because "
        "correlated repeats break the binomial\nassumption behind them in the "
        "anti-conservative direction."
    )

    if result.run_dir is not None:
        written = save_snapshot_report(report, Path(result.run_dir))
        for name, path in written.items():
            print(f"Saved {name} snapshot metrics to {path}")
        print(f"Run directory: {result.run_dir}")


if __name__ == "__main__":
    main()
