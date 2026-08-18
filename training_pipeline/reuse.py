"""Recover the hyperparameters a past run found, without re-tuning.

Every run already persists its Optuna trials (``optuna_selected_trial.json``
and ``optuna_best_trial.json``), so a study that took hours never has to be
repeated just to reuse what it discovered. This module reads those artifacts
back and renders them as a YAML block you can paste into an experiment file
under ``optuna.fixed_params``, which makes the next run skip tuning entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training_pipeline.tuning import NON_XGB_TRIAL_PARAMS, USE_SAMPLE_WEIGHT_PARAM


@dataclass
class RunHyperparameters:
    """Hyperparameters recovered from a saved experiment run."""

    run_dir: Path
    #: XGBoost parameters only -- training-protocol params are split out below.
    params: dict[str, Any]
    n_estimators: int
    sample_weight_lambda: float | None
    #: "selected" (lexicographic pick) or "best" (lowest CV MAE).
    source: str
    trial_number: int | None
    cv_mae: float | None = None
    #: Set instead of cv_mae for classifier runs, whose trial value is log loss.
    cv_logloss: float | None = None
    cv_ou_acc: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_yaml_block(self) -> str:
        """A ready-to-paste ``optuna:`` block that skips tuning."""
        lines = [
            f"# Recovered from {self.run_dir.name}"
            f" (trial {self.trial_number}, {self.source}).",
        ]
        if self.cv_logloss is not None:
            summary = f"# CV log loss {self.cv_logloss:.5f}"
            if self.cv_ou_acc is not None:
                summary += f", CV OU accuracy {self.cv_ou_acc:.2%}"
            lines.append(summary)
        elif self.cv_mae is not None:
            summary = f"# CV MAE {self.cv_mae:.4f}"
            if self.cv_ou_acc is not None:
                summary += f", CV OU accuracy {self.cv_ou_acc:.2%}"
            lines.append(summary)
        lines += [
            "optuna:",
            "  fixed_params:",
        ]
        for key, value in sorted(self.params.items()):
            lines.append(f"    {key}: {value!r}")
        lines.append(f"  fixed_n_estimators: {self.n_estimators}")
        if self.sample_weight_lambda is not None:
            lines.append(
                f"  fixed_sample_weight_lambda: {self.sample_weight_lambda!r}"
            )
        return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_run_hyperparameters(
    run_dir: str | Path, *, prefer_selected: bool = True
) -> RunHyperparameters:
    """Load the hyperparameters a run settled on.

    Prefers the lexicographically *selected* trial (best MAE within tolerance,
    then best OU accuracy) over the raw lowest-MAE trial, because that is the
    one the run actually used. Pass ``prefer_selected=False`` to force the
    lowest-MAE trial instead.
    """
    run_dir = Path(run_dir)
    selected = _read_json(run_dir / "optuna_selected_trial.json")
    best = _read_json(run_dir / "optuna_best_trial.json")

    payload: dict[str, Any] | None = None
    source = ""
    if prefer_selected and selected:
        payload, source = selected.get("selected_trial"), "selected"
    if payload is None and best:
        payload, source = best.get("best_trial"), "best"
    if payload is None and selected:
        payload, source = selected.get("selected_trial"), "selected"

    if not payload:
        raise FileNotFoundError(
            f"No Optuna trial artifacts found in {run_dir}. The run may have "
            "used optuna.fixed_params (nothing to recover), or been saved "
            "without experiment artifacts."
        )

    trial_params = dict(payload.get("params") or {})
    user_attrs = dict(payload.get("user_attrs") or {})

    params = {k: v for k, v in trial_params.items() if k not in NON_XGB_TRIAL_PARAMS}
    lambda_ = trial_params.get("sample_weight_lambda") or user_attrs.get(
        "sample_weight_lambda"
    )
    # Respect a trial that decided against weighting altogether.
    if trial_params.get(USE_SAMPLE_WEIGHT_PARAM) is False:
        lambda_ = None

    n_estimators = (
        user_attrs.get("median_best_iteration")
        or user_attrs.get("mean_best_iteration")
        or trial_params.get("n_estimators")
    )
    if not n_estimators:
        raise ValueError(
            f"Could not determine n_estimators from {run_dir}: the trial "
            "recorded neither median_best_iteration nor mean_best_iteration."
        )

    return RunHyperparameters(
        run_dir=run_dir,
        params=params,
        n_estimators=max(50, int(round(float(n_estimators)))),
        sample_weight_lambda=float(lambda_) if lambda_ is not None else None,
        source=source,
        trial_number=payload.get("number"),
        # No blind fallback to payload["value"]: for a classifier that value is
        # log loss, and filing it under cv_mae would print "CV MAE 0.6931" in
        # the recovered-hyperparameter block.
        cv_mae=user_attrs.get("mean_mae"),
        cv_logloss=user_attrs.get("mean_logloss"),
        cv_ou_acc=user_attrs.get("mean_ou_acc"),
        extras=user_attrs,
    )


def find_best_run_hyperparameters(
    root_dir: str | Path = Path("artifacts/experiments"),
    *,
    rank_by: str = "roi",
) -> RunHyperparameters:
    """Hyperparameters from the best run on the leaderboard.

    Defaults to ranking by ROI, i.e. "the run that actually made the most
    money", rather than by CV MAE.
    """
    from training_pipeline.leaderboard import build_leaderboard

    leaderboard = build_leaderboard(root_dir, sort_by=rank_by)
    if leaderboard.empty:
        raise FileNotFoundError(f"No runs found under {root_dir}.")

    root = Path(root_dir)
    for run_name in leaderboard["run_name"]:
        try:
            return load_run_hyperparameters(root / str(run_name))
        except (FileNotFoundError, ValueError):
            continue
    raise FileNotFoundError(
        f"No run under {root_dir} has recoverable Optuna hyperparameters."
    )
