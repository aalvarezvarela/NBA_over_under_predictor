"""Turn a pile of runs into controlled, one-variable-at-a-time comparisons.

The survey notebook asks "what is in this set of runs?". This module answers the
harder question underneath it: **"what is the effect of changing X?"** -- which
is only answerable between runs that are identical apart from X.

The method is deliberately mechanical rather than curated. Every run's config is
reduced to a small vector of *factors* (strategy, window, overtime handling,
missingness budget, ...). Two runs form a contrast on factor ``X`` when their
vectors agree on every factor except ``X``. Nothing is matched by name, by
campaign, or by the ``hypothesis`` string, because all three are prose written
before the run existed and none of them is checked against what actually ran.

Two consequences worth knowing:

- **A run that changed two things at once appears in no contrast.** That is the
  point. ``line_error_4500_maxna_200_no_consensus`` differs from the 4500
  baseline in both the missingness budget and the dropped columns, so it can
  only be read against the ``maxna_200`` cell, and it is -- as a contrast on
  ``drop_consensus`` alone.
- **Factors are read from ``config.json``, never from the run's name.** Names
  have been wrong before: ``line_error_3750_anchor`` is not a new
  configuration, it is a byte-for-byte replication of ``line_error_3750``, and
  only the config shows that.

Cohort comparability is separate and stricter. ROI measured over 89 holdout
games is not a worse measurement of the same thing as ROI over 416 -- it is a
measurement of a different period. :func:`flag_cohorts` marks those runs so
they can be shown and excluded rather than silently averaged in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from training_pipeline.reporting.loaders import load_config_flat

#: One column per experimental knob that has actually been varied across runs,
#: mapped from its dotted path in ``config.json``. Anything not listed here is
#: assumed constant; if a new campaign varies something else, add it here or
#: its runs will be matched as though they were identical.
FACTOR_SOURCES: dict[str, str] = {
    "train_games": "walk_forward.train_games",
    "max_folds": "walk_forward.max_folds",
    "wf_strategy": "walk_forward.strategy",
    "season_floor": "data.season_year_floor",
    "max_na_per_row": "cleaning.max_na_per_row",
    "nan_threshold": "cleaning.nan_threshold",
    "n_trials": "optuna.n_trials",
    # --- rolling-origin CV and the tuned training protocol -------------------
    # These change what a trial IS, so a run before and after any of them is not
    # the same experiment. Without them here the summary notebook would match a
    # rolling-origin run against a test-anchored one as though they were
    # identical and quietly average the two.
    "retrain_every_days": "walk_forward.retrain_every_days",
    "eval_span_games": "walk_forward.eval_span_games",
    "objective_aggregation": "optuna.objective_aggregation",
    "tune_n_estimators": "optuna.tune_n_estimators",
    "pruner_warmup_fraction": "optuna.pruner_warmup_fraction",
    #: Strength of the planted diagnostic signal. The only thing that varies
    #: across the planted-signal campaign, so without it those four runs
    #: match as identical and the comparison they exist for disappears.
    "planted_variance": "diagnostics.planted_signal.variance_explained",
}

#: Factors needing a rule rather than a straight read; see ``_derived_factors``.
DERIVED_FACTORS: tuple[str, ...] = (
    "strategy",
    "data_build",
    "exclude_overtime",
    "keep_playoffs",
    "drop_consensus",
    "sample_weighting",
    "train_games_tuned",
    "n_estimators_range",
    "is_diagnostic",
)

FACTOR_COLUMNS: tuple[str, ...] = (*DERIVED_FACTORS, *FACTOR_SOURCES)

#: Metrics every strategy reports, so a contrast table can carry the same
#: columns whichever factor it varies. MAE and log-loss are deliberately absent:
#: neither exists for all three strategies.
CONTRAST_METRICS: tuple[str, ...] = (
    "cv_win_rate",
    "cv_n_bets",
    "cv_roi",
    "win_rate",
    "n_bets",
    "roi",
    "bet_rate",
    "seed_roi_range",
)


def _render_range(value: Any) -> str:
    """A hashable, readable rendering of an IntRange/FloatRange config block.

    ``design_matrix`` groups on factor values, so an unhashable dict would raise
    rather than produce a wrong answer -- but a stringified dict would also
    reorder between pandas versions. Reading the three fields explicitly keeps
    the label stable.
    """
    if not isinstance(value, dict):
        return "" if value is None else str(value)
    low, high = value.get("low"), value.get("high")
    scale = "log" if value.get("log") else "lin"
    return f"{low}-{high}:{scale}"


def _derived_factors(config: dict[str, Any], row: Any) -> dict[str, Any]:
    """The factors that are not a single field read straight out of the config.

    ``exclude_overtime_from_training`` postdates the earliest runs, so its
    absence means the filter did not exist yet, i.e. False -- not unknown. The
    same reasoning applies to ``sample_weight.enabled``.
    """
    csv_path = str(config.get("data.csv_path", ""))
    excluded = str(config.get("cleaning.exclude_cols_containing", ""))
    return {
        # prepare_runs has already back-filled this from target_family for runs
        # saved before prediction_strategy existed.
        "strategy": row.get("prediction_strategy"),
        # The training CSV identifies the feature build; the file name is the
        # only thing that distinguishes them in every run, old and new.
        "data_build": "old" if Path(csv_path).name.startswith("old_") else "2.0",
        "exclude_overtime": bool(
            config.get("data.exclude_overtime_from_training", False)
        ),
        "keep_playoffs": not bool(config.get("data.exclude_playoffs", True)),
        "drop_consensus": "consensus" in excluded,
        "sample_weighting": bool(config.get("sample_weight.enabled", False)),
        # A run that TUNED the window is not comparable with one that fixed it,
        # even when train_games happens to read the same: the fixed run's value
        # is what it used, the tuned run's is only a fallback. Without this flag
        # the two would land in the same cell of the design matrix.
        "train_games_tuned": bool(
            config.get("walk_forward.train_games_choices") or False
        ),
        # The search range itself, so widening it forks the comparison. Rendered
        # as text because a dict is not hashable and groupby needs to hash it.
        "n_estimators_range": _render_range(
            config.get("optuna.search_space.n_estimators_range")
        ),
        # A planted-signal run must never be matched against a real one: its
        # metrics are inflated by a target-derived feature by construction.
        # Absent from every config written before diagnostics existed, which
        # means False -- those runs predate any way to enable it.
        "is_diagnostic": bool(
            config.get("diagnostics.planted_signal.enabled", False)
        ),
    }


def design_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    """Add one column per factor, read from each run's saved config.

    Returns the runs frame unchanged apart from the added factor columns and a
    ``design`` column: the factor vector rendered as text, which is what makes
    "these two runs are the same experiment" visible at a glance.
    """
    factor_rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        config = load_config_flat(row)
        values = _derived_factors(config, row)
        for name, source in FACTOR_SOURCES.items():
            values[name] = config.get(source)
        factor_rows.append(values)

    factors = pd.DataFrame(factor_rows, index=runs.index)
    out = pd.concat([runs.drop(columns=list(factors.columns), errors="ignore"),
                     factors], axis=1)
    out["design"] = [
        " | ".join(f"{name}={values[name]}" for name in FACTOR_COLUMNS)
        for _, values in factors.iterrows()
    ]
    return out


#: How a deviation from the set's usual value is written in a label. A callable
#: receives the deviating value; anything else is used as a literal tag.
_DEVIATION_TAGS: dict[str, Any] = {
    "exclude_overtime": "no-OT",
    "keep_playoffs": "+playoffs",
    "drop_consensus": "no-consensus",
    "sample_weighting": "weighted",
    "is_diagnostic": "DIAGNOSTIC",
    "planted_variance": lambda value: f"planted{value:.3f}",
    "data_build": lambda value: f"{value}-data",
    "max_na_per_row": lambda value: f"maxna{value:.0f}",
    "nan_threshold": lambda value: f"nanthr{value:.0f}",
    "max_folds": lambda value: f"{value:.0f}folds",
    "n_trials": lambda value: f"{value:.0f}trials",
    "season_floor": lambda value: f"from{value:.0f}",
    "wf_strategy": lambda value: str(value),
    "retrain_every_days": lambda value: f"every{value:.0f}d",
    "eval_span_games": lambda value: f"span{value:.0f}",
    "objective_aggregation": lambda value: str(value),
    "tune_n_estimators": "tuned-rounds",
    "train_games_tuned": "tuned-window",
    "n_estimators_range": lambda value: f"rounds{value}",
    "pruner_warmup_fraction": lambda value: f"warmup{value:.2f}",
}


def describe_labels(runs: pd.DataFrame) -> pd.DataFrame:
    """Replace ``label`` with one that says what makes each run different.

    ``prepare_runs`` labels a run ``strategy · window`` and disambiguates
    collisions with four characters of its experiment id, which is unique but
    tells you nothing: half the set reads ``line_error · 3750 [9afa]``. Here the
    suffix is the run's actual deviations from what the rest of the set does --
    ``line_error · 3750 · no-OT`` -- so a table can be read without going back
    to the config.

    "What the rest of the set does" is the modal value of each factor across
    the runs passed in, so a label describes a run *relative to its cohort*
    rather than to a hard-coded baseline that would go stale. Filter first,
    then label.
    """
    view = design_matrix(runs) if "design" not in runs.columns else runs.copy()

    tags: list[list[str]] = [[] for _ in range(len(view))]
    for factor, render in _DEVIATION_TAGS.items():
        if factor not in view.columns:
            continue
        values = view[factor]
        known = values.dropna()
        if known.empty:
            continue
        usual = known.mode().iat[0]
        for position, value in enumerate(values):
            if pd.isna(value) or value == usual:
                continue
            tags[position].append(render(value) if callable(render) else str(render))

    base = (
        view["strategy_short"]
        + " · "
        + view["train_games"].astype("Int64").astype(str)
    )
    view["label"] = [
        " · ".join([name, *suffix]) if suffix else name
        for name, suffix in zip(base, tags, strict=True)
    ]

    # Two runs of the *same* configuration deviate identically, so labelling
    # cannot separate them. Number them rather than letting a groupby merge two
    # runs into one row.
    duplicated = view["label"].duplicated(keep=False)
    if duplicated.any():
        order = view.groupby("label").cumcount() + 1
        view.loc[duplicated, "label"] = (
            view.loc[duplicated, "label"] + " (run " + order[duplicated].astype(str) + ")"
        )
    return view


def flag_cohorts(runs: pd.DataFrame) -> pd.DataFrame:
    """Mark runs whose holdout is not the period everything else was scored on.

    Keyed on the holdout's **start and end dates**, not on its game count. The
    count is the wrong test in both directions: the classifier scores 409 games
    where the regressors score 416 over the very same two months -- seven games
    it could not price, not a different period -- while a run that keeps
    playoffs collapses to 89 games precisely *because* its window moved. Dates
    separate those two cases; a count threshold conflates them.

    The reference cohort is the most common window in the set. Runs outside it
    keep their metrics and get a note saying which period they measured, so
    they can be displayed and excluded rather than quietly averaged in.
    """
    out = runs.copy()
    window = (
        out["holdout_start"].astype(str).str[:10]
        + " → "
        + out["holdout_end"].astype(str).str[:10]
    )
    out["holdout_window"] = window

    known = window[out["holdout_start"].notna()]
    if known.empty:
        out["cohort_ok"] = False
        out["cohort_note"] = "no holdout window recorded"
        return out

    reference = str(known.mode().iat[0])
    out["cohort_ok"] = window.eq(reference) & out["holdout_start"].notna()
    out["cohort_note"] = out.apply(
        lambda row: (
            "—" if row["cohort_ok"] else f"scored on {row['holdout_window']}"
        ),
        axis=1,
    )
    out.attrs["reference_cohort"] = reference
    return out


def contrasts(
    runs: pd.DataFrame,
    factor: str,
    *,
    factors: tuple[str, ...] = FACTOR_COLUMNS,
    metrics: tuple[str, ...] = CONTRAST_METRICS,
    require_cohort: bool = True,
    collapse_replicates: bool = True,
) -> pd.DataFrame:
    """Every set of runs differing *only* in ``factor``.

    Groups on all factors except ``factor`` and keeps groups that contain more
    than one level of it. The result is long -- one row per run -- with a
    ``contrast`` column naming the group, so a table or a chart can be faceted
    on it without re-deriving the matching.

    ``require_cohort`` drops runs flagged by :func:`flag_cohorts` first, which
    is nearly always right: a matched pair scored on different periods is
    matched on the wrong thing.

    ``collapse_replicates`` keeps only the earliest run of any configuration
    that was executed more than once. Without it a replicated cell appears
    twice at the same level and :func:`effect` reads the pair as a change from
    a level to itself. The replicates are not lost -- :func:`replicates`
    reports them, which is where they belong, since what they measure is noise
    rather than an effect.
    """
    if factor not in factors:
        raise KeyError(
            f"{factor!r} is not a factor. Known factors: {sorted(factors)}"
        )

    view = design_matrix(runs) if "design" not in runs.columns else runs.copy()
    if require_cohort:
        if "cohort_ok" not in view.columns:
            view = flag_cohorts(view)
        view = view[view["cohort_ok"]]
    if collapse_replicates:
        view = view.sort_values("created_at").drop_duplicates("design", keep="first")
    if view.empty:
        return view

    others = [name for name in factors if name != factor]
    # NaN is a real level here (a factor absent from an older config), and
    # dropna=False keeps those groups instead of deleting them without a word.
    grouped = view.groupby(others, dropna=False, sort=False)

    kept: list[tuple[tuple, pd.DataFrame]] = []
    for key, group in grouped:
        if group[factor].nunique(dropna=False) >= 2:
            kept.append((key, group.copy()))

    if not kept:
        return view.iloc[0:0]

    # Name each contrast by the held-constant factors that actually *differ
    # between* the contrasts on screen. Spelling out all eleven every time
    # buries the one or two that distinguish "line_error at 3750" from
    # "line_error at 4500" under a wall of identical text.
    # NaN must be normalised before the comparison. A factor absent from every
    # config (a knob added after those runs) yields a distinct NaN per group, and
    # distinct NaNs do not collapse in a set -- so the factor would count as
    # "varying" and be spelled out in every contrast label, burying the one
    # factor that actually differs. This bit only shows up once a new factor is
    # added, which is exactly when the labels matter most.
    def _level(value: Any) -> Any:
        return "<absent>" if value is None or pd.isna(value) else value

    varying = [
        name
        for position, name in enumerate(others)
        if len({_level(key[position]) for key, _ in kept}) > 1
    ]
    positions = {name: others.index(name) for name in varying}
    for key, group in kept:
        group["contrast"] = (
            " | ".join(f"{name}={key[position]}" for name, position in positions.items())
            if varying else "all runs"
        )

    columns = [
        "contrast", "label", factor, *metrics,
        "prediction_strategy", "strategy_short", "run_name",
    ]
    out = pd.concat([group for _, group in kept])
    return (
        out[[c for c in columns if c in out.columns]]
        .sort_values(["contrast", factor])
        .reset_index(drop=True)
    )


def effect(
    contrast_table: pd.DataFrame,
    factor: str,
    *,
    metrics: tuple[str, ...] = ("cv_win_rate", "win_rate", "cv_roi", "roi"),
) -> pd.DataFrame:
    """Change in each metric relative to the first level within each contrast.

    Only meaningful for an ordered or clearly-baselined factor -- "first level"
    is the lowest window, or False before True. For a many-level factor read
    the contrast table itself rather than these deltas.

    The deltas are reported next to ``seed_roi_range`` on purpose: an ROI delta
    smaller than a single run's own spread across seeds is not an effect.
    """
    if contrast_table.empty:
        return contrast_table

    rows: list[dict[str, Any]] = []
    for contrast, group in contrast_table.groupby("contrast", sort=False):
        group = group.sort_values(factor)
        baseline = group.iloc[0]
        for _, row in group.iloc[1:].iterrows():
            record: dict[str, Any] = {
                "contrast": contrast,
                "from": baseline[factor],
                "to": row[factor],
                "baseline_run": baseline["label"],
                "variant_run": row["label"],
            }
            for metric in metrics:
                if metric in group.columns:
                    record[f"d_{metric}"] = row[metric] - baseline[metric]
            record["seed_roi_range"] = max(
                float(baseline.get("seed_roi_range") or 0.0),
                float(row.get("seed_roi_range") or 0.0),
            )
            if "d_roi" in record and pd.notna(record["d_roi"]):
                record["beats_seed_noise"] = (
                    abs(record["d_roi"]) > record["seed_roi_range"]
                )
            rows.append(record)

    return pd.DataFrame(rows)


def replicates(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs sharing a full factor vector -- the same experiment, run twice.

    These are the empirical floor on how much a number can move for no reason
    at all, and they are worth more than any single comparison in the set:
    every effect below their spread is indistinguishable from having run the
    same thing again.
    """
    view = design_matrix(runs) if "design" not in runs.columns else runs
    duplicated = view["design"].duplicated(keep=False)
    columns = ["design", "label", "run_name", *CONTRAST_METRICS]
    return (
        view[duplicated][[c for c in columns if c in view.columns]]
        .sort_values(["design", "label"])
        .reset_index(drop=True)
    )
