"""Cross-run comparison: reads every saved experiment run and builds one
leaderboard row per run, always framed relative to the "trust the bookmaker
line" baseline (never in isolation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from training_pipeline.config import TargetFamily
from training_pipeline.tracking import parse_run_dir_timestamp

DEFAULT_EXPERIMENT_ROOT = Path("artifacts") / "experiments"


def discover_run_dirs(root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    """Load whatever artifacts exist for one run directory into a flat summary
    row. Missing files degrade to None rather than raising, so a partially
    populated run (e.g. one that skipped experiment tracking) still shows up.
    """
    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / "metadata.json") or {}
    config = _read_json(run_dir / "config.json") or {}
    final_metrics = _read_json(run_dir / "final_test_metrics.json") or {}
    baseline_metrics = _read_json(run_dir / "baseline_metrics.json") or {}
    betting_metrics = _read_json(run_dir / "betting_metrics.json") or {}
    selected_trial_payload = _read_json(run_dir / "optuna_selected_trial.json")
    best_trial_payload = _read_json(run_dir / "optuna_best_trial.json")

    trial: dict[str, Any] | None = None
    if selected_trial_payload:
        trial = selected_trial_payload.get("selected_trial")
    elif best_trial_payload:
        trial = best_trial_payload.get("best_trial")

    cv = final_metrics.get("cv") or {}
    holdout = final_metrics.get("holdout") or {}
    baseline_cv = baseline_metrics.get("cv") or {}
    baseline_holdout = baseline_metrics.get("holdout") or {}
    optuna_cfg = config.get("optuna") or {}
    walk_forward_cfg = config.get("walk_forward") or {}
    refit_cfg = config.get("refit") or {}
    betting_primary = betting_metrics.get("primary") or {}
    bias_baseline = betting_metrics.get("baseline_bias_corrected") or {}
    bias_baseline_betting = betting_metrics.get("baseline_bias_corrected_betting") or {}

    # metadata.json holds *resolved* labels and the run timestamp; config.json
    # holds the raw (possibly None) user-supplied values. Prefer metadata, fall
    # back to config, and finally to the timestamp embedded in the directory
    # name so runs written before created_at was recorded still sort correctly.
    created_at = metadata.get("created_at")
    if not created_at:
        parsed = parse_run_dir_timestamp(run_dir)
        created_at = parsed.isoformat() if parsed is not None else None

    return {
        "run_name": run_dir.name,
        "experiment_id": metadata.get("experiment_id"),
        "experiment_name": metadata.get("experiment_name"),
        # Runs sharing a comparison_group are the ones actually meant to be
        # read against each other.
        "comparison_group": (
            metadata.get("comparison_group") or config.get("comparison_group")
        ),
        "hypothesis": metadata.get("hypothesis") or config.get("hypothesis"),
        "tags": metadata.get("tags") or config.get("tags"),
        "data_version": metadata.get("data_version"),
        "dataset_checksum": metadata.get("dataset_checksum"),
        # Manually curated label; falls back to config.json for runs whose
        # metadata predates it being recorded there.
        "training_version": (
            metadata.get("training_version") or config.get("training_version")
        ),
        "created_at": created_at,
        "target_family": metadata.get("target_family") or config.get("target_family"),
        "window_dir_label": (
            metadata.get("window_dir_label") or config.get("window_dir_label")
        ),
        "config_fingerprint": metadata.get("config_fingerprint"),
        # --- evaluation cohort: what this run's numbers were measured on ---
        # ROI and MAE are only comparable between runs scored on the same
        # holdout window and a similar number of games, so these travel next
        # to the metrics instead of being left implicit.
        "holdout_start": metadata.get("holdout_start"),
        "holdout_end": metadata.get("holdout_end"),
        "holdout_n_games": metadata.get("holdout_n_games"),
        "n_candidates": betting_primary.get("n_candidates"),
        # walk_forward is the single source of truth; refit_cfg is only read
        # as a fallback for runs saved before the two were unified.
        "train_games": (
            walk_forward_cfg.get("train_games") or refit_cfg.get("train_games")
        ),
        "n_trials": optuna_cfg.get("n_trials"),
        "objective_name": optuna_cfg.get("objective_name"),
        "cv_mae": cv.get("mae"),
        "cv_rmse": cv.get("rmse"),
        "cv_ou_acc": cv.get("ou_acc"),
        "final_test_mae": holdout.get("mae"),
        "final_test_rmse": holdout.get("rmse"),
        "final_test_ou_acc": holdout.get("ou_acc"),
        "baseline_cv_mae": baseline_cv.get("mae"),
        "baseline_cv_rmse": baseline_cv.get("rmse"),
        "baseline_holdout_mae": baseline_holdout.get("mae"),
        "baseline_holdout_rmse": baseline_holdout.get("rmse"),
        # --- profit-oriented metrics at the primary edge threshold ---
        "bet_min_edge": betting_primary.get("min_edge"),
        "n_bets": betting_primary.get("n_bets"),
        "bet_rate": betting_primary.get("bet_rate"),
        "win_rate": betting_primary.get("win_rate"),
        "win_rate_ci_low": betting_primary.get("win_rate_ci_low"),
        "win_rate_ci_high": betting_primary.get("win_rate_ci_high"),
        "break_even_rate": betting_primary.get("break_even_rate"),
        "edge_vs_break_even": betting_primary.get("edge_vs_break_even"),
        "roi": betting_primary.get("roi"),
        "profit_units": betting_primary.get("profit_units"),
        "is_significant": betting_primary.get("is_significant"),
        # --- the harder "line + historical drift" null ---
        "bias_baseline_mae": bias_baseline.get("mae"),
        "bias_baseline_roi": bias_baseline_betting.get("roi"),
        "dev_line_error_bias": betting_metrics.get("dev_line_error_bias"),
        "selected_trial_number": (trial or {}).get("number"),
    }


#: Columns worth looking at first when comparing runs. ROI leads because it is
#: the only column that answers "would this have made money"; n_bets and
#: is_significant sit next to it because an ROI computed on 20 bets is noise.
#: The holdout window and candidate count follow immediately, because a ranking
#: only means something between runs measured on the same evaluation cohort --
#: they are shown rather than enforced, since comparing configurations is the
#: whole point of a leaderboard.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "comparison_group",
    "training_version",
    "target_family",
    "window_dir_label",
    "roi",
    "n_bets",
    "win_rate",
    "break_even_rate",
    "edge_vs_break_even",
    "is_significant",
    "holdout_start",
    "holdout_end",
    "n_candidates",
    "config_fingerprint",
    "bias_baseline_roi",
    "final_test_mae",
    "baseline_holdout_mae",
    "mae_improvement_over_baseline_pct",
    "created_at",
)


def build_leaderboard(
    root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    target_family: TargetFamily | str | None = None,
    sort_by: str = "roi",
) -> pd.DataFrame:
    """One row per saved run, ranked by betting profitability.

    Sorted by ROI rather than MAE improvement by default: against a closing
    total line, MAE differences between runs are ~0.1 points and are dominated
    by noise, whereas ROI (with n_bets and is_significant alongside it) is the
    quantity that actually decides whether a model is worth using. The MAE
    columns are still reported for diagnostics.
    """
    rows = [load_run_summary(run_dir) for run_dir in discover_run_dirs(root_dir)]

    if target_family is not None:
        target_value = (
            target_family.value if isinstance(target_family, TargetFamily) else target_family
        )
        rows = [row for row in rows if row.get("target_family") == target_value]

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["mae_improvement_over_baseline_pct"] = (
        df["baseline_holdout_mae"] - df["final_test_mae"]
    ) / df["baseline_holdout_mae"]
    df["rmse_improvement_over_baseline_pct"] = (
        df["baseline_holdout_rmse"] - df["final_test_rmse"]
    ) / df["baseline_holdout_rmse"]
    df["beats_baseline_mae"] = df["mae_improvement_over_baseline_pct"] > 0
    df["beats_baseline_rmse"] = df["rmse_improvement_over_baseline_pct"] > 0
    df["roi_vs_bias_baseline"] = df["roi"] - df["bias_baseline_roi"]

    if sort_by not in df.columns:
        raise KeyError(
            f"sort_by={sort_by!r} is not a leaderboard column. Available: "
            f"{sorted(df.columns)}"
        )

    return df.sort_values(sort_by, ascending=False, na_position="last").reset_index(
        drop=True
    )


def headline_leaderboard(
    root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    target_family: TargetFamily | str | None = None,
) -> pd.DataFrame:
    """``build_leaderboard`` trimmed to HEADLINE_COLUMNS for quick display."""
    df = build_leaderboard(root_dir, target_family=target_family)
    if df.empty:
        return df
    return df[[column for column in HEADLINE_COLUMNS if column in df.columns]]
