"""Check a campaign is runnable BEFORE committing hours of GPU time to it.

    poetry run python scripts/preflight_campaign.py experiments/window_overtime_2026_08

Exits non-zero if anything would block the campaign, so a runner can gate on it.

The check that justifies this script's existence is the training-window one.
``make_test_anchored_walk_forward_splits`` selects a fold's training rows with
``tail(train_games)`` and only skips the fold when fewer than
``min_train_games`` remain. So a window larger than the data supports does NOT
raise -- the early folds quietly train on fewer games than requested and the
run completes looking healthy, having silently stopped being the comparison it
was designed to be. That ceiling depends on the row count AFTER cleaning, which
is not knowable from the CSV without doing the cleaning, which is why guessing
it from raw row counts got it wrong twice.

Everything here is measured, never assumed: it builds the real splits with the
real code and reports the actual per-fold training sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_pipeline.cli import load_config  # noqa: E402
from training_pipeline.config import ExperimentConfig  # noqa: E402
from training_pipeline.data import (  # noqa: E402
    compute_file_checksum,
    prepare_dataset,
    training_eligible_mask,
)
from training_pipeline.splits import (  # noqa: E402
    build_holdout_split,
    build_walk_forward_splits,
)

OK, WARN, FAIL = "ok", "warn", "FAIL"


def _data_fingerprint(config: ExperimentConfig) -> str:
    """Configs sharing this share a cleaned dataset, so it is prepared once.

    Cleaning a ~250MB CSV is the expensive step; a campaign typically has far
    fewer distinct data settings than configs.
    """
    payload = {
        "data": config.data.model_dump(mode="json"),
        "cleaning": config.cleaning.model_dump(mode="json"),
        "family": config.family.value,
        "line_col": config.line_col,
    }
    return json.dumps(payload, sort_keys=True)


def _prepare_quietly(config: ExperimentConfig) -> tuple[Any, Any, Any]:
    """prepare_dataset + holdout split with the cleaning report suppressed."""
    payload = config.model_dump()
    payload["cleaning"]["verbose"] = 0
    quiet = ExperimentConfig.model_validate(payload)
    with redirect_stdout(StringIO()):
        prepared = prepare_dataset(quiet)
        df_dev, df_test = build_holdout_split(prepared.df_full, quiet)
    return prepared, df_dev, df_test


def check_campaign(campaign_dir: Path, *, skip_data: bool) -> int:
    configs_paths = sorted(p for p in campaign_dir.glob("*.y*ml") if not p.name.startswith("_"))
    if not configs_paths:
        print(f"No experiment configs found in {campaign_dir}")
        return 1

    print(f"Pre-flight: {campaign_dir}  ({len(configs_paths)} configs)\n")
    problems: list[str] = []

    # --- 1. every config parses ---------------------------------------------
    configs: dict[str, ExperimentConfig] = {}
    for path in configs_paths:
        try:
            configs[path.stem] = load_config(path)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the check
            problems.append(f"{path.name}: does not load -- {type(exc).__name__}: {exc}")
            print(f"  {FAIL}  {path.name}: {type(exc).__name__}: {exc}")
    if not configs:
        return 1
    print(f"  {OK}    all {len(configs)} configs parse\n")

    # --- 2. datasets exist, and their bytes are the pinned ones -------------
    print("Datasets")
    seen: dict[Path, str] = {}
    for name, config in configs.items():
        csv = Path(config.data.csv_path)
        csv = csv if csv.is_absolute() else REPO_ROOT / csv
        if not csv.exists():
            problems.append(f"{name}: dataset missing -- {csv}")
            print(f"  {FAIL}  {name}: missing {csv}")
            continue
        if csv not in seen:
            seen[csv] = compute_file_checksum(csv)
        actual, pinned = seen[csv], config.data.expected_checksum
        if pinned is None:
            print(f"  {WARN}  {name}: no expected_checksum pinned (actual {actual})")
        elif pinned != actual:
            problems.append(f"{name}: checksum mismatch, pinned {pinned} got {actual}")
            print(f"  {FAIL}  {name}: checksum {pinned} != {actual}")
        else:
            print(f"  {OK}    {name}: {csv.name} matches {actual}")
    print()

    # --- 3. the design matrix -----------------------------------------------
    print("Design matrix (fields that differ across the campaign)")
    flat = {
        name: {
            "strategy": c.strategy.value,
            "train_games": c.walk_forward.train_games,
            "csv": Path(c.data.csv_path).name,
            "no_overtime": c.data.exclude_overtime_from_training,
            "drop_playoffs": c.data.exclude_playoffs,
            "max_na_per_row": c.cleaning.max_na_per_row,
            "nan_threshold": c.cleaning.nan_threshold,
            "exclude_cols": str(c.cleaning.exclude_cols_containing),
            "n_trials": c.optuna.n_trials,
            "seeds": str(list(c.evaluation_seeds)),
        }
        for name, c in configs.items()
    }
    varying = [
        key for key in next(iter(flat.values()))
        if len({str(v[key]) for v in flat.values()}) > 1
    ]
    if not varying:
        print("  (every config is identical -- nothing is being compared)")
    else:
        width = max(len(n) for n in flat)
        widths = {
            k: max(len(k), max(len(str(v[k])) for v in flat.values())) for k in varying
        }
        header = "  ".join(f"{k:<{widths[k]}}" for k in varying)
        print(f"  {'config':<{width}}  {header}")
        for name, row in flat.items():
            cells = "  ".join(f"{str(row[k]):<{widths[k]}}" for k in varying)
            print(f"  {name:<{width}}  {cells}")
    print()

    if not configs.values() or all(not c.evaluation_seeds for c in configs.values()):
        print(f"  {WARN}  no config sets evaluation_seeds: results will have no error bar\n")

    if skip_data:
        print("Skipping the data-dependent window check (--skip-data).")
        return _verdict(problems, checked_windows=False)

    # --- 4. does each training window actually fit? -------------------------
    print("Training window feasibility (cleaning each distinct dataset once)")
    groups: dict[str, list[str]] = {}
    for name, config in configs.items():
        groups.setdefault(_data_fingerprint(config), []).append(name)

    for members in groups.values():
        config = configs[members[0]]
        try:
            prepared, df_dev, df_test = _prepare_quietly(config)
        except Exception as exc:  # noqa: BLE001
            for name in members:
                problems.append(f"{name}: prepare_dataset failed -- {exc}")
                print(f"  {FAIL}  {name}: prepare_dataset raised {type(exc).__name__}: {exc}")
            continue

        eligible = int(training_eligible_mask(df_dev, config).sum())
        print(f"  {Path(config.data.csv_path).name}"
              f"  (playoffs {'kept' if not config.data.exclude_playoffs else 'dropped'},"
              f" max_na={config.cleaning.max_na_per_row})")
        print(f"     cleaned={len(prepared.df_full)}  dev={len(df_dev)}  "
              f"holdout={len(df_test)}  train-eligible={eligible}")

        for name in members:
            member = configs[name]
            requested = member.walk_forward.train_games
            # Feasibility must be measured with training-row filters OFF.
            # build_walk_forward_splits applies them (overtime, etc.) AFTER
            # tail(train_games), so a filtered fold is legitimately smaller than
            # the window -- that is the filter working, not the window
            # overflowing. Only an unfiltered shortfall means the data does not
            # reach back far enough.
            unfiltered = member.model_dump()
            unfiltered["data"]["exclude_overtime_from_training"] = False
            unfiltered["cleaning"]["verbose"] = 0
            probe = ExperimentConfig.model_validate(unfiltered)
            try:
                with redirect_stdout(StringIO()):
                    splits, _ = build_walk_forward_splits(df_dev, probe)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: split building failed -- {exc}")
                print(f"     {FAIL}  {name}: {type(exc).__name__}: {exc}")
                continue

            sizes = [len(train) for train, _ in splits]
            n_folds = len(splits)
            short = [s for s in sizes if requested and s < requested]
            expected_folds = member.walk_forward.max_folds

            if requested and short:
                largest_ok = min(short)
                problems.append(
                    f"{name}: {len(short)} of {n_folds} folds cannot reach "
                    f"train_games={requested} (smallest pool {largest_ok}). The window "
                    "does not fit, and this does NOT raise at runtime -- those folds "
                    f"would quietly train on less. Largest window all folds support: "
                    f"~{largest_ok}."
                )
                print(f"     {FAIL}  {name}: {len(short)}/{n_folds} folds short "
                      f"(min {largest_ok} vs requested {requested}) "
                      f"-> max feasible ~{largest_ok}")
            elif n_folds < expected_folds:
                problems.append(
                    f"{name}: only {n_folds} folds built, {expected_folds} configured -- "
                    "fold counts must match for runs to be comparable."
                )
                print(f"     {FAIL}  {name}: {n_folds} folds, expected {expected_folds}")
            else:
                note = ""
                if member.data.exclude_overtime_from_training:
                    with redirect_stdout(StringIO()):
                        filtered, _ = build_walk_forward_splits(df_dev, member)
                    actual = min(len(t) for t, _ in filtered)
                    note = (f"; training filter then removes rows, leaving "
                            f"{actual}/fold (expected, not a misfit)")
                print(f"     {OK}    {name}: {n_folds} folds, all reach "
                      f"{requested}{note}")
        print()

    return _verdict(problems, checked_windows=True)


def _verdict(problems: list[str], *, checked_windows: bool) -> int:
    if problems:
        print(f"{len(problems)} blocking problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if checked_windows:
        print("Pre-flight passed: configs, checksums and training windows all check out.")
    else:
        # Saying "passed" here would imply the one check that actually needs
        # doing, and the one that silently misbehaves when wrong.
        print("Configs and checksums check out. The training-window check was SKIPPED, "
              "so a window too large for the data would still go unnoticed -- rerun "
              "without --skip-data before committing the campaign.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("campaign_dir", type=Path,
                        help="Folder of experiment YAML files, e.g. experiments/<campaign>")
    parser.add_argument("--skip-data", action="store_true",
                        help="Config and checksum checks only; skips cleaning, which is "
                             "the slow part but also the only way to verify the window.")
    args = parser.parse_args()

    campaign = args.campaign_dir
    campaign = campaign if campaign.is_absolute() else REPO_ROOT / campaign
    if not campaign.is_dir():
        print(f"Not a directory: {campaign}")
        return 1
    return check_campaign(campaign, skip_data=args.skip_data)


if __name__ == "__main__":
    raise SystemExit(main())
