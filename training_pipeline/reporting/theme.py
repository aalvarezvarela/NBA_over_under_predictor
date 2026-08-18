"""Shared visual language for the reporting notebooks.

One fixed colour per prediction strategy, assigned in slot order and never
cycled, so a strategy keeps its identity no matter which runs a filter leaves
on screen. The three slots are validated to stay distinguishable under
colour-vision deficiency across every pair, which is what makes them safe for
scatter and small-multiple charts as well as bars.

The bookmaker line is deliberately grey rather than taking a fourth hue: it is
a reference the models are measured against, not a competing series.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

STRATEGY_COLOR: dict[str, str] = {
    "total_points_regressor": "#2a78d6",   # blue
    "line_error_regressor": "#eb6834",     # orange
    "over_under_classifier": "#1baf7a",    # aqua
}
STRATEGY_SHORT: dict[str, str] = {
    "total_points_regressor": "total_points",
    "line_error_regressor": "line_error",
    "over_under_classifier": "classifier",
}
#: Runs saved before ``prediction_strategy`` existed carry only a target family.
FAMILY_TO_STRATEGY: dict[str, str] = {
    "total_points": "total_points_regressor",
    "line_error": "line_error_regressor",
    "over_under": "over_under_classifier",
}

LINE_REF = "#898781"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

#: -110 American odds, the standard price on NBA totals.
DECIMAL_ODDS = 1.0 + 100.0 / 110.0
BREAK_EVEN = 1.0 / DECIMAL_ODDS


def apply_theme() -> None:
    """Recessive grid, muted axes, ink-coloured text. Call once per notebook."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 7,
    })


def colour_of(row: Any) -> str:
    return STRATEGY_COLOR.get(row["prediction_strategy"], MUTED)


def short_name(strategy: str) -> str:
    return STRATEGY_SHORT.get(strategy, strategy)


def strategy_legend(ax: plt.Axes, strategies: Any, extra: list[Line2D] | None = None) -> None:
    """A legend is always present for two or more series, so identity never
    rests on colour alone."""
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=9,
               color=STRATEGY_COLOR.get(s, MUTED), label=short_name(s))
        for s in strategies
    ]
    handles += extra or []
    if len(handles) >= 2:
        ax.legend(handles=handles, fontsize=8, loc="best")


def label_bars(ax: plt.Axes, bars: Any, values: Any, fmt: str = "{:.1%}") -> None:
    """Direct value labels on bars.

    Required relief for the lower-contrast palette slot, and it lets every bar
    be read without tracking back to the axis.
    """
    for bar, value in zip(bars, values, strict=True):
        if pd.isna(value):
            continue
        height = bar.get_height()
        ax.annotate(
            fmt.format(value), (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points", xytext=(0, 4 if height >= 0 else -11),
            ha="center", fontsize=8, color=INK_2,
        )


def rotate_xticks(ax: plt.Axes, rotation: int = 20) -> None:
    ax.tick_params(axis="x", rotation=rotation)
    for tick in ax.get_xticklabels():
        # set_horizontalalignment, not the set_ha alias: the alias is not on
        # the typed Text stub, so mypy rejects it.
        tick.set_horizontalalignment("right")


def percent_frame(df: pd.DataFrame, percent: Any = (), decimals: Any = ()) -> Any:
    """Style a table: percentages as %, chosen columns to 3 decimals."""
    formatters = {c: "{:.2%}" for c in percent if c in df.columns}
    formatters.update({c: "{:.3f}" for c in decimals if c in df.columns})
    return df.style.format(formatters, na_rep="—")


#: Config fields that can distinguish one run from another, with how each
#: renders as a label token. Order here is the order tokens appear.
#:
#: A token is only added when the field actually VARIES across the runs being
#: compared, and only for runs that differ from the most common value. So a
#: label says what is unusual about a run rather than restating the campaign's
#: shared settings -- "line_error · 4500 · na200" reads as "the 4500 cell with
#: the relaxed row cap", which is the question that cell exists to answer.
LABEL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("data.csv_path", lambda v: "old-data" if "old_" in str(v) else "2.0-data"),
    ("data.exclude_overtime_from_training", lambda v: "no-OT" if v else "with-OT"),
    ("data.exclude_playoffs", lambda v: "no-playoffs" if v else "+playoffs"),
    ("cleaning.max_na_per_row", lambda v: f"na{v}"),
    ("cleaning.exclude_cols_containing",
     lambda v: "no-consensus" if v and "consensus_pct" in str(v) else "std-cols"),
    ("cleaning.nan_threshold", lambda v: f"nanthr{v:g}"),
    ("optuna.n_trials", lambda v: f"{v}trials"),
)


def _label_tokens(configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Per run, the tokens describing how it departs from the common setup."""
    tokens: dict[str, list[str]] = {name: [] for name in configs}
    for field, render in LABEL_FIELDS:
        values = {name: cfg.get(field) for name, cfg in configs.items()}
        present = {v for v in values.values() if v is not None}
        if len(present) < 2:
            continue  # not a discriminator among these runs
        counts = pd.Series([str(v) for v in values.values()]).value_counts()
        baseline = counts.index[0]
        for name, value in values.items():
            if value is not None and str(value) != baseline:
                tokens[name].append(str(render(value)))
    return tokens


def prepare_runs(runs: pd.DataFrame, *, describe: bool = True) -> pd.DataFrame:
    """Add the strategy/label columns every chart keys off.

    Labels start from strategy and training window, then gain a token for each
    config field that differs from the rest of the comparison (see
    LABEL_FIELDS) -- so runs from a campaign that varies overtime, playoffs or
    the missing-data cap are told apart by name instead of by a hash.

    ``describe=False`` falls back to strategy + window only, which is enough
    when every run differs on those alone.
    """
    runs = runs.copy()
    runs["prediction_strategy"] = runs["prediction_strategy"].fillna(
        runs["target_family"].map(FAMILY_TO_STRATEGY)
    )
    runs["strategy_short"] = runs["prediction_strategy"].map(STRATEGY_SHORT).fillna(
        runs["prediction_strategy"]
    )
    runs["is_classifier"] = runs["prediction_strategy"] == "over_under_classifier"

    runs["label"] = (
        runs["strategy_short"] + " · " + runs["train_games"].astype("Int64").astype(str)
    )

    if describe:
        from training_pipeline.reporting import loaders

        configs = {
            str(row["run_name"]): loaders.load_config_flat(row)
            for _, row in runs.iterrows()
        }
        configs = {k: v for k, v in configs.items() if v}
        if configs:
            tokens = _label_tokens(configs)
            runs["label"] += runs["run_name"].map(
                lambda name: "".join(f" · {t}" for t in tokens.get(str(name), []))
            ).fillna("")

    if runs["source_root"].nunique() > 1:
        runs["label"] += " · " + runs["source_root"]

    # Last resort. Reaching here means two runs are genuinely indistinguishable
    # by configuration -- a repeat, or a difference in a field not listed above.
    duplicated = runs["label"].duplicated(keep=False)
    if duplicated.any():
        runs.loc[duplicated, "label"] = (
            runs.loc[duplicated, "label"]
            + " [" + runs.loc[duplicated, "experiment_id"].astype(str).str[:4] + "]"
        )

    return runs.sort_values(
        ["prediction_strategy", "train_games", "created_at"]
    ).reset_index(drop=True)
