from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.optuna_total_points import objective_total_points_mae
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

DEFAULT_EXPERIMENT_ROOT = Path("artifacts") / "experiments"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return slug or "experiment"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return path


def create_experiment_run_dir(
    experiment_name: str,
    *,
    root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT,
    timestamp: datetime | None = None,
) -> Path:
    timestamp = timestamp or datetime.now(tz=UTC)
    run_name = f"{_slugify(experiment_name)}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def optuna_sqlite_storage_url(
    run_dir: str | Path, filename: str = "optuna_study.db"
) -> str:
    sqlite_path = Path(run_dir) / filename
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path.resolve().as_posix()}"


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".json":
        df.to_json(path, orient="records", date_format="iso", indent=2)
    else:
        raise ValueError(f"Unsupported dataframe artifact extension: {path.suffix}")

    return path


def _trial_payload(trial: Any) -> dict[str, Any] | None:
    if trial is None:
        return None

    return {
        "number": trial.number,
        "value": trial.value,
        "state": trial.state.name,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def tune_xgb_total_points_optuna_persistent(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    line_col: str,
    storage: str,
    n_trials: int = 80,
    timeout: int | None = None,
    objective_name: str = "reg:squarederror",
    study_name: str = "xgb_total_points_mae",
    load_if_exists: bool = True,
) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=16),
        pruner=MedianPruner(n_warmup_steps=5),
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        lambda trial: objective_total_points_mae(
            trial,
            X=X,
            y=y,
            splits=splits,
            line_col=line_col,
            objective_name=objective_name,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=True,
    )
    return study


def save_experiment_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    return write_json(Path(run_dir) / "metadata.json", metadata)


def save_feature_schema(run_dir: str | Path, feature_names: list[str]) -> Path:
    return write_json(
        Path(run_dir) / "feature_schema.json",
        {"feature_names": feature_names, "n_features": len(feature_names)},
    )


def save_optuna_artifacts(
    *,
    run_dir: str | Path,
    study: Any,
    trials_df: pd.DataFrame | None = None,
    selected_trial: Any | None = None,
    candidates_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    artifacts: dict[str, Path] = {}

    if trials_df is None:
        trials_df = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs")
        )

    artifacts["trials"] = save_dataframe(trials_df, run_dir / "optuna_trials.csv")
    artifacts["best_trial"] = write_json(
        run_dir / "optuna_best_trial.json",
        {"best_trial": _trial_payload(study.best_trial)},
    )

    if selected_trial is not None:
        artifacts["selected_trial"] = write_json(
            run_dir / "optuna_selected_trial.json",
            {"selected_trial": _trial_payload(selected_trial)},
        )

    if candidates_df is not None:
        artifacts["candidates"] = save_dataframe(
            candidates_df, run_dir / "optuna_lexicographic_candidates.csv"
        )

    return artifacts


def save_final_test_artifacts(
    *,
    run_dir: str | Path,
    metrics: dict[str, Any],
    predictions_df: pd.DataFrame,
    threshold_results_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    artifacts = {
        "metrics": write_json(run_dir / "final_test_metrics.json", metrics),
        "predictions": save_dataframe(
            predictions_df, run_dir / "final_test_predictions.parquet"
        ),
    }

    if threshold_results_df is not None:
        artifacts["thresholds"] = save_dataframe(
            threshold_results_df, run_dir / "threshold_results.csv"
        )

    return artifacts


def save_walk_forward_artifacts(
    *,
    run_dir: str | Path,
    daily_results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "daily": save_dataframe(daily_results_df, run_dir / "walk_forward_daily.csv"),
        "predictions": save_dataframe(
            predictions_df, run_dir / "walk_forward_predictions.parquet"
        ),
    }
