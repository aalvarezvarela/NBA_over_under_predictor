"""The figures the survey notebook draws.

Each function takes the prepared runs frame, loads whatever artifacts it needs,
draws one figure, and returns the data it computed so the notebook can narrate
from the same numbers it plotted rather than recomputing them.

Form choices worth knowing about, because they are not arbitrary:

- Values that differ by ~0.1 on a base of ~14 (model MAE versus the line) are
  drawn as a dumbbell, never as bars. A bar encodes magnitude by its length, so
  showing that difference as bars would need a truncated axis, and a truncated
  bar misleads. A dot encodes position, where a non-zero axis is honest.
- Per-run curves are faceted by strategy rather than overlaid. Two runs of the
  same strategy share a colour, so a single panel would make identity ambiguous
  and turn six crossing lines into spaghetti.
- The seed-range rule is a plain vertical line, not a capped errorbar: it is an
  observed min-max range, not a symmetric confidence interval, and should not
  be dressed as one.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from training_pipeline.reporting import loaders
from training_pipeline.reporting.theme import (
    AXIS,
    BREAK_EVEN,
    CRITICAL,
    DECIMAL_ODDS,
    GOOD,
    GRID,
    INK,
    INK_2,
    LINE_REF,
    MUTED,
    STRATEGY_COLOR,
    SURFACE,
    colour_of,
    label_bars,
    rotate_xticks,
    short_name,
    strategy_legend,
)

#: Edge ranges in points. Fine at the bottom, where the question "is a 1-point
#: edge better than a 0.5-point one?" actually lives, and coarser in the tail
#: where large edges are rare and half-point bins would hold single digits.
DEFAULT_EDGE_BINS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, np.inf)

#: Below this a bin's win rate is too noisy to read as anything.
MIN_BIN_SIZE = 30


# ------------------------------------------------------- CV vs final test
def plot_cv_vs_test_win_rate(runs: pd.DataFrame) -> pd.DataFrame:
    """Win rate in cross-validation against win rate on the held-out period.

    Both are measured at each run's own bet threshold, so this is the
    like-for-like version of "did it hold up?". CV chose the hyperparameters
    and so flatters the model; the holdout did not but is far smaller. The
    dumbbell makes the direction and size of the move the thing you read.
    """
    view = runs.dropna(subset=["cv_win_rate", "win_rate"]).copy()
    if view.empty:
        return view

    view["win_rate_drop"] = view["cv_win_rate"] - view["win_rate"]
    view = view.sort_values("win_rate", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(view) + 3))
    y = np.arange(len(view))

    for position, (_, row) in enumerate(view.iterrows()):
        colour = colour_of(row)
        ax.plot([row["cv_win_rate"], row["win_rate"]], [position, position],
                color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        # Hollow = cross-validation, filled = the holdout it has to survive.
        ax.plot([row["cv_win_rate"]], [position], marker="o", markersize=10,
                markerfacecolor=SURFACE, markeredgecolor=colour,
                markeredgewidth=2.2, zorder=3)
        ax.plot([row["win_rate"]], [position], marker="o", markersize=10,
                color=colour, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate(
            f"{row['cv_win_rate']:.1%} → {row['win_rate']:.1%}"
            f"   ({int(row['cv_n_bets'])} → {int(row['n_bets'])} bets)",
            (max(row["cv_win_rate"], row["win_rate"]), position),
            textcoords="offset points", xytext=(10, -3), fontsize=8, color=INK_2,
        )

    ax.axvline(BREAK_EVEN, color=CRITICAL, linestyle="--", linewidth=1.5, zorder=2)
    # Top of the axis, not the bottom: the bottom row's own annotation starts
    # just right of the break-even line and the two collide there.
    ax.annotate("break-even 52.38%", xy=(BREAK_EVEN, 1),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=CRITICAL)
    ax.set_yticks(y, view["label"])
    ax.set_xlabel("Win rate")
    ax.set_title("Cross-validation → held-out test, at each run's own bet threshold")
    ax.set_xlim(0.35, 0.80)
    ax.invert_yaxis()
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor=SURFACE,
               markeredgecolor=MUTED, markeredgewidth=2.2, label="cross-validation"),
        Line2D([], [], marker="o", linestyle="", markersize=10, color=MUTED,
               label="held-out test"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()

    return view[[
        "label", "prediction_strategy", "holdout_evaluation",
        "cv_win_rate", "cv_n_bets", "win_rate", "n_bets", "win_rate_drop",
        "cv_roi", "roi",
    ]]


# ------------------------------------------------------- edge ranges
def edge_range_breakdown(
    runs: pd.DataFrame,
    *,
    bins: tuple[float, ...] = DEFAULT_EDGE_BINS,
    regressors_only: bool = True,
) -> pd.DataFrame:
    """Win rate by how far the prediction sat from the line.

    The premise behind every minimum-edge threshold is that a bigger predicted
    edge is a better bet. This tests it directly, in points, on every candidate
    game rather than only the ones that cleared the threshold -- otherwise the
    filter's own effect is baked into the answer.

    Restricted to regressors by default: their ``predicted_edge`` is points
    away from the line, whereas a classifier's is a difference of expected
    values, so pooling the two would add unlike quantities.
    """
    from training_pipeline.betting import wilson_interval

    selected = runs[~runs["is_classifier"]] if regressors_only else runs
    labels = [
        (f"{low:g}–{high:g}" if np.isfinite(high) else f"{low:g}+")
        for low, high in zip(bins[:-1], bins[1:], strict=True)
    ]

    rows: list[dict[str, Any]] = []
    for _, run in selected.iterrows():
        threshold = run["bet_min_edge"] if pd.notna(run["bet_min_edge"]) else 0.0
        for source, frame in loaders.load_all_predictions(run):
            frame = frame.assign(abs_edge=frame["predicted_edge"].abs())
            frame["edge_range"] = pd.cut(frame["abs_edge"], list(bins), right=False,
                                         labels=labels, include_lowest=True)
            grouped = frame.groupby("edge_range", observed=False).agg(
                n_bets=("won", "size"), n_wins=("won", "sum"))
            total_games = int(grouped["n_bets"].sum())
            # Games at or above each range's floor, walking down from the top:
            # this is exactly the volume a threshold set there would keep.
            at_or_above = grouped["n_bets"][::-1].cumsum()[::-1]
            wins_at_or_above = grouped["n_wins"][::-1].cumsum()[::-1]

            for index, (edge_range, stats) in enumerate(grouped.iterrows()):
                n_bets, n_wins = int(stats["n_bets"]), int(stats["n_wins"])
                low = bins[index]
                ci_low, ci_high = (
                    wilson_interval(n_wins, n_bets) if n_bets else (np.nan, np.nan)
                )
                kept = int(at_or_above.iloc[index])
                kept_wins = int(wins_at_or_above.iloc[index])
                rows.append({
                    "label": run["label"],
                    "prediction_strategy": run["prediction_strategy"],
                    "source": source,
                    "edge_range": str(edge_range),
                    "edge_low": low,
                    "n_bets": n_bets,
                    "n_wins": n_wins,
                    "win_rate": n_wins / n_bets if n_bets else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    # Share of this run's games landing in this range.
                    "pct_of_games": n_bets / total_games if total_games else np.nan,
                    # And what a threshold at this range's floor would keep --
                    # the number to weigh a win rate against before adopting it.
                    "pct_at_or_above": kept / total_games if total_games else np.nan,
                    "n_at_or_above": kept,
                    "win_rate_at_or_above": kept_wins / kept if kept else np.nan,
                    # Whether this run would actually have taken these bets.
                    "currently_bet": low >= threshold,
                    "too_few_to_read": n_bets < MIN_BIN_SIZE,
                })
    return pd.DataFrame(rows)


#: Row order for the combined per-run table.
COMBINED_SOURCE_ORDER = ("cross-validation", "holdout", "% of games")


def combined_edge_table(
    breakdown: pd.DataFrame, *, cumulative: bool = True
) -> pd.DataFrame:
    """One row block per run: both win rates and the volume, stacked.

    Reads as a single table rather than two held side by side: for each column
    you get the cross-validation win rate, the holdout win rate, and the share
    of that run's games involved -- which is what stops a 75% cell being read
    as a result when it rests on four bets.

    ``cumulative`` (the default) makes each column **every game at or above
    that edge**, which is the operational question: a threshold is a floor, not
    a band, so ">= 1 point" is the population you would actually bet. It also
    keeps the columns well populated -- a disjoint 4-6 band can hold a handful
    of games and swing wildly, while ">= 4" at least accumulates the tail.
    Set it False for disjoint bands, which answer the different question of
    where in the range the wins actually sit.

    ``% of games`` pools both prediction sets. Their distributions are close
    but not identical (up to ~5 points apart on these runs), so the row answers
    "roughly how much of this model's output qualifies", not "exactly what
    share of the holdout". Every value is a percentage, so the rows share one
    format.
    """
    if breakdown.empty:
        return breakdown

    rate_column = "win_rate_at_or_above" if cumulative else "win_rate"
    frame = breakdown.copy()
    frame["column"] = (
        frame["edge_low"].map(lambda low: f"≥{low:g}") if cumulative
        else frame["edge_range"]
    )
    order = frame.sort_values("edge_low")["column"].drop_duplicates().tolist()

    rates = frame[["label", "source", "column", "edge_low", rate_column]].rename(
        columns={rate_column: "value"}
    )

    # Volume pooled across sources, then accumulated downward when cumulative
    # so it matches the win rate beside it.
    volume = (
        frame.groupby(["label", "column", "edge_low"], observed=True)
        .agg(n_bets=("n_bets", "sum"))
        .reset_index()
        .sort_values(["label", "edge_low"])
    )
    parts = []
    for _, group in volume.groupby("label", sort=False):
        group = group.sort_values("edge_low").copy()
        total = group["n_bets"].sum()
        counted = group["n_bets"][::-1].cumsum()[::-1] if cumulative else group["n_bets"]
        group["value"] = counted / total if total else np.nan
        parts.append(group)
    volume = pd.concat(parts)
    volume["source"] = "% of games"

    combined = pd.concat(
        [rates, volume[["label", "source", "column", "edge_low", "value"]]],
        ignore_index=True,
    )
    combined["source"] = pd.Categorical(
        combined["source"], categories=COMBINED_SOURCE_ORDER, ordered=True
    )
    combined["column"] = pd.Categorical(combined["column"], categories=order, ordered=True)
    return combined.sort_values(["label", "source", "edge_low"]).reset_index(drop=True)


def pool_edge_ranges(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Pool the per-run breakdown across runs, per source.

    Pooling is what makes the bins readable: one run's 0.5–1.0 bucket holds a
    few dozen games, which cannot separate a 55% win rate from a 50% one.
    """
    from training_pipeline.betting import wilson_interval

    if breakdown.empty:
        return breakdown

    pooled = (
        breakdown.groupby(["source", "edge_range", "edge_low"], observed=True)
        .agg(n_bets=("n_bets", "sum"), n_wins=("n_wins", "sum"),
             runs=("label", "nunique"))
        .reset_index()
        .sort_values(["source", "edge_low"])
    )
    pooled["win_rate"] = pooled["n_wins"] / pooled["n_bets"].replace(0, np.nan)
    intervals = [
        wilson_interval(int(w), int(n)) if n else (np.nan, np.nan)
        for w, n in zip(pooled["n_wins"], pooled["n_bets"], strict=True)
    ]
    pooled["ci_low"] = [low for low, _ in intervals]
    pooled["ci_high"] = [high for _, high in intervals]
    pooled["too_few_to_read"] = pooled["n_bets"] < MIN_BIN_SIZE

    # Volume, recomputed within each source rather than carried through the
    # groupby: the pooled totals differ from any single run's.
    parts = []
    for _, group in pooled.groupby("source", sort=False):
        group = group.sort_values("edge_low").copy()
        total = group["n_bets"].sum()
        kept = group["n_bets"][::-1].cumsum()[::-1]
        kept_wins = group["n_wins"][::-1].cumsum()[::-1]
        group["pct_of_games"] = group["n_bets"] / total if total else np.nan
        group["pct_at_or_above"] = kept / total if total else np.nan
        group["n_at_or_above"] = kept
        group["win_rate_at_or_above"] = kept_wins / kept.replace(0, np.nan)
        parts.append(group)

    return pd.concat(parts).reset_index(drop=True)


def plot_edge_ranges(pooled: pd.DataFrame) -> None:
    """Win rate and volume by edge range, cross-validation beside holdout."""
    if pooled.empty:
        return

    sources = [s for s in ("cross-validation", "holdout") if s in set(pooled["source"])]
    order = pooled.sort_values("edge_low")["edge_range"].drop_duplicates().tolist()
    positions = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    styles = {"cross-validation": ("-", "o"), "holdout": ("--", "s")}

    for source in sources:
        subset = pooled[pooled["source"] == source].set_index("edge_range").reindex(order)
        linestyle, marker = styles.get(source, ("-", "o"))
        # One hue per source, and the two sources are the only series here, so
        # the first two categorical slots are the right choice.
        colour = STRATEGY_COLOR["total_points_regressor"] if source == "cross-validation" \
            else STRATEGY_COLOR["line_error_regressor"]

        axes[0].plot(positions, subset["win_rate"], marker=marker, linestyle=linestyle,
                     color=colour, markeredgecolor=SURFACE, markeredgewidth=1.2,
                     label=source, zorder=3)
        # Interval per point: a bin of 20 games and one of 400 look identical
        # without it, and only one of them means anything.
        for position, (_, row) in enumerate(subset.iterrows()):
            if pd.isna(row["win_rate"]):
                continue
            axes[0].plot([position, position], [row["ci_low"], row["ci_high"]],
                         color=colour, linewidth=1.6, alpha=0.45, zorder=2)

        axes[1].plot(positions, subset["n_bets"], marker=marker, linestyle=linestyle,
                     color=colour, markeredgecolor=SURFACE, markeredgewidth=1.2,
                     label=source, zorder=3)

    axes[0].axhline(BREAK_EVEN, color=CRITICAL, linestyle="--", linewidth=1.5, zorder=1)
    axes[0].annotate("break-even 52.38%", xy=(1, BREAK_EVEN),
                     xycoords=("axes fraction", "data"), textcoords="offset points",
                     xytext=(-4, 5), ha="right", fontsize=8, color=CRITICAL)
    axes[0].set_ylabel("Win rate")
    axes[0].set_title("Win rate by predicted edge\n(bars = 95% interval; wide = too few games)")
    axes[1].set_ylabel("Bets in range")
    axes[1].set_title("How many games fall in each range")
    for ax in axes:
        ax.set_xticks(positions, order)
        ax.set_xlabel("|predicted total − line|  (points)")
        ax.legend(fontsize=8)

    fig.suptitle("Does a bigger predicted edge win more often?  (regressors only)",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------- portfolio
def plot_portfolio(runs: pd.DataFrame) -> pd.Series:
    """Coverage per strategy, and when each run happened."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))

    per_strategy = runs["prediction_strategy"].value_counts()
    bars = axes[0].bar(
        [short_name(s) for s in per_strategy.index], per_strategy.values,
        color=[STRATEGY_COLOR.get(s, MUTED) for s in per_strategy.index], width=0.6,
    )
    label_bars(axes[0], bars, per_strategy.values, fmt="{:.0f}")
    axes[0].set_title("Runs per strategy")
    axes[0].set_ylabel("Runs")
    axes[0].set_ylim(0, per_strategy.max() * 1.25)

    for strategy, group in runs.groupby("prediction_strategy"):
        axes[1].scatter(group["created_at"], group["train_games"],
                        color=STRATEGY_COLOR.get(strategy, MUTED), s=90,
                        edgecolor=SURFACE, linewidth=1.5, zorder=3)
    axes[1].set_title("When each run happened, and at which window")
    axes[1].set_ylabel("Training window (games)")
    axes[1].set_xlabel("Run timestamp")
    rotate_xticks(axes[1])
    strategy_legend(axes[1], runs["prediction_strategy"].unique())

    plt.tight_layout()
    plt.show()
    return per_strategy


def plot_selectivity(runs: pd.DataFrame) -> None:
    """Bet rate -- the one selectivity measure that means the same thing for
    every strategy, since the threshold itself does not."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bars = ax.bar(runs["label"], runs["bet_rate"],
                  color=[colour_of(r) for _, r in runs.iterrows()], width=0.62)
    label_bars(ax, bars, runs["bet_rate"])
    ax.set_title("How choosy each run is — share of candidate games actually bet")
    ax.set_ylabel("Bet rate")
    ax.set_ylim(0, min(1.0, float(runs["bet_rate"].max()) * 1.3))
    rotate_xticks(ax, 25)
    strategy_legend(ax, runs["prediction_strategy"].unique())
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------- headline
def plot_roi_with_seed_noise(runs: pd.DataFrame) -> None:
    """Holdout ROI with the seed range each configuration produced."""
    fig, ax = plt.subplots(figsize=(12.5, 5))
    x = np.arange(len(runs))
    ax.bar(x, runs["roi"], color=[colour_of(r) for _, r in runs.iterrows()],
           width=0.6, zorder=2)

    above, below = [], []
    for position, (_, row) in enumerate(runs.iterrows()):
        top = bottom = row["roi"]
        if pd.notna(row.get("seed_roi_range")):
            half = row["seed_roi_range"] / 2
            top, bottom = row["roi"] + half, row["roi"] - half
            ax.plot([position, position], [bottom, top], color=INK, linewidth=2,
                    zorder=4, solid_capstyle="butt")
            ax.plot([position], [row["roi"]], marker="_", markersize=13,
                    color=INK, zorder=5)
        above.append(top)
        below.append(bottom)

    # Label clear of the seed rule -- at the bar top the rule runs straight
    # through the text and the value becomes unreadable.
    for position, (_, row) in enumerate(runs.iterrows()):
        if pd.isna(row["roi"]):
            continue
        positive = row["roi"] >= 0
        anchor = above[position] if positive else below[position]
        ax.annotate(f"{row['roi']:.1%}", (position, anchor),
                    textcoords="offset points", xytext=(0, 6 if positive else -13),
                    ha="center", fontsize=8, color=INK_2, zorder=6)

    margin = (max(above) - min(below)) * 0.12
    ax.set_ylim(min(min(below), 0) - margin, max(above) + margin)
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    ax.set_title("Holdout ROI, with the seed-noise range each configuration produces")
    ax.set_ylabel("ROI")
    ax.set_xticks(x, runs["label"], rotation=25, ha="right")
    strategy_legend(ax, runs["prediction_strategy"].unique(),
                    extra=[Line2D([], [], color=INK, linewidth=2, label="seed range")])
    plt.tight_layout()
    plt.show()


def plot_significance_and_volume(runs: pd.DataFrame) -> pd.DataFrame:
    """Win-rate intervals against break-even, and the volume behind them."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order = runs.sort_values("win_rate", ascending=False).reset_index(drop=True)
    y = np.arange(len(order))

    for position, (_, row) in enumerate(order.iterrows()):
        if pd.isna(row["win_rate"]):
            continue
        colour = colour_of(row)
        axes[0].plot([row["win_rate_ci_low"], row["win_rate_ci_high"]],
                     [position, position], color=colour, linewidth=3,
                     solid_capstyle="round", zorder=3, alpha=0.55)
        axes[0].plot([row["win_rate"]], [position], marker="o", markersize=9,
                     color=colour, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        proven = pd.notna(row["win_rate_ci_low"]) and row["win_rate_ci_low"] > BREAK_EVEN
        axes[0].annotate(
            f"{row['win_rate']:.1%}  ({int(row['n_bets'])} bets)" + ("  ✓" if proven else ""),
            (max(row["win_rate_ci_high"], BREAK_EVEN), position),
            textcoords="offset points", xytext=(8, -3), fontsize=8,
            color=GOOD if proven else INK_2,
        )

    axes[0].axvline(BREAK_EVEN, color=CRITICAL, linestyle="--", linewidth=1.5, zorder=2)
    # Anchored in axes coordinates: this axis is inverted, so a data-space
    # position at the end of the list lands off the bottom edge and is clipped.
    axes[0].annotate("break-even 52.38%", xy=(BREAK_EVEN, 0),
                     xycoords=("data", "axes fraction"), textcoords="offset points",
                     xytext=(6, 8), fontsize=8, color=CRITICAL)
    axes[0].set_yticks(y, order["label"])
    axes[0].set_xlabel("Win rate (95% Wilson interval)")
    axes[0].set_title("Is the edge distinguishable from break-even?")
    axes[0].set_xlim(0.30, 0.85)
    axes[0].invert_yaxis()

    width = 0.38
    for offset, (column, alpha, name) in enumerate(
        ((("n_bets"), 1.0, "holdout"), (("cv_n_bets"), 0.45, "cross-validation"))
    ):
        shift = (offset - 0.5) * width
        bars = axes[1].barh(y + shift, order[column], width,
                            color=[colour_of(r) for _, r in order.iterrows()],
                            alpha=alpha, label=name)
        for bar, value in zip(bars, order[column], strict=True):
            if pd.isna(value):
                continue
            axes[1].annotate(f"{int(value)}",
                             (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                             textcoords="offset points", xytext=(4, -3),
                             fontsize=7.5, color=INK_2)
    axes[1].set_yticks(y, order["label"])
    axes[1].set_xlabel("Bets placed")
    axes[1].set_title("Bet volume — the constraint on proving anything")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    return order


# --------------------------------------------------------------- cv vs holdout
def plot_cv_vs_holdout(runs: pd.DataFrame) -> pd.DataFrame:
    have_both = runs.dropna(subset=["cv_roi", "roi"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if have_both.empty:
        axes[0].text(0.5, 0.5, "No run has both CV and holdout ROI", ha="center",
                     va="center", transform=axes[0].transAxes, color=MUTED)
        plt.tight_layout()
        plt.show()
        return have_both

    limit = float(have_both[["cv_roi", "roi"]].abs().max().max()) * 1.25
    axes[0].plot([-limit, limit], [-limit, limit], color=AXIS, linewidth=1,
                 linestyle="--", zorder=1)
    axes[0].axhline(0, color=AXIS, linewidth=1, zorder=1)
    axes[0].axvline(0, color=AXIS, linewidth=1, zorder=1)
    for _, row in have_both.iterrows():
        axes[0].scatter(row["cv_roi"], row["roi"], s=130, color=colour_of(row),
                        edgecolor=SURFACE, linewidth=1.5, zorder=3)
        axes[0].annotate(row["label"], (row["cv_roi"], row["roi"]),
                         textcoords="offset points", xytext=(9, 4),
                         fontsize=7.5, color=INK_2)
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_xlabel("CV ROI (biased — chose the hyperparameters)")
    axes[0].set_ylabel("Holdout ROI (honest — but few bets)")
    axes[0].set_title("Agreement between the two measurements\n(on the dashed line = they agree)")
    strategy_legend(axes[0], have_both["prediction_strategy"].unique())

    gap = have_both.assign(gap=have_both["cv_roi"] - have_both["roi"]).sort_values("gap")
    bars = axes[1].barh(np.arange(len(gap)), gap["gap"],
                        color=[colour_of(r) for _, r in gap.iterrows()], height=0.6)
    axes[1].axvline(0, color=AXIS, linewidth=1.2)
    axes[1].set_yticks(np.arange(len(gap)), gap["label"])
    axes[1].set_xlabel("CV ROI − holdout ROI")
    axes[1].set_title("Positive = CV flattered the model\nNegative = the holdout got lucky")
    for bar, value in zip(bars, gap["gap"], strict=True):
        axes[1].annotate(f"{value:+.1%}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                         textcoords="offset points",
                         xytext=(5 if value >= 0 else -34, -3), fontsize=8, color=INK_2)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame([{
        "label": row["label"], "cv_roi": row["cv_roi"], "holdout_roi": row["roi"],
        "gap": row["cv_roi"] - row["roi"],
        "same_sign": np.sign(row["cv_roi"]) == np.sign(row["roi"]),
        "cv_bets": row["cv_n_bets"], "holdout_bets": row["n_bets"],
        "cv_profitable_folds": row.get("cv_n_profitable_folds"),
    } for _, row in have_both.iterrows()])


# --------------------------------------------------------------- strategy
def plot_strategy_window(runs: pd.DataFrame) -> pd.DataFrame:
    panels = [
        (runs.pivot_table(index="train_games", columns="prediction_strategy",
                          values="roi", aggfunc="mean"), "Holdout ROI", "{:.1%}"),
        (runs.pivot_table(index="train_games", columns="prediction_strategy",
                          values="cv_roi", aggfunc="mean"), "CV ROI", "{:.1%}"),
        (runs.pivot_table(index="train_games", columns="prediction_strategy",
                          values="n_bets", aggfunc="mean"), "Bets placed (holdout)", "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    for ax, (table, title, fmt) in zip(axes, panels, strict=True):
        if table.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
            ax.set_title(title)
            continue
        windows = table.index.astype(int).astype(str).tolist()
        positions = np.arange(len(windows))
        strategies = list(table.columns)
        width = 0.8 / max(len(strategies), 1)
        for offset, strategy in enumerate(strategies):
            shift = (offset - (len(strategies) - 1) / 2) * width
            values = table[strategy].to_numpy(dtype=float)
            bars = ax.bar(positions + shift, values, width * 0.9,
                          color=STRATEGY_COLOR.get(strategy, MUTED))
            for bar, value in zip(bars, values, strict=True):
                if np.isnan(value):
                    continue
                ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                            textcoords="offset points",
                            xytext=(0, 3 if value >= 0 else -10),
                            ha="center", fontsize=7.5, color=INK_2)
        if "ROI" in title:
            ax.axhline(0, color=AXIS, linewidth=1.2)
        ax.set_xticks(positions, windows)
        ax.set_xlabel("Training window (games)")
        ax.set_title(title)

    axes[0].set_ylabel("ROI")
    strategy_legend(axes[0], list(panels[0][0].columns))
    fig.suptitle("Strategy × training window", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    return (
        runs.groupby("strategy_short")
        .agg(runs=("label", "count"), mean_roi=("roi", "mean"),
             best_roi=("roi", "max"), worst_roi=("roi", "min"),
             mean_cv_roi=("cv_roi", "mean"), mean_bets=("n_bets", "mean"),
             mean_bet_rate=("bet_rate", "mean"), mean_win_rate=("win_rate", "mean"))
        .reset_index()
    )


# --------------------------------------------------------------- regressors
def plot_regressor_accuracy(runs: pd.DataFrame) -> pd.DataFrame | None:
    regressors = runs[~runs["is_classifier"]].dropna(subset=["final_test_mae"])
    if regressors.empty:
        return None

    view = regressors.assign(
        holdout_edge=regressors["baseline_holdout_mae"] - regressors["final_test_mae"],
        cv_edge=regressors["baseline_cv_mae"] - regressors["cv_mae"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    positions = np.arange(len(view))

    # A dumbbell, not paired bars -- see the module docstring.
    for position, (_, row) in enumerate(view.iterrows()):
        model_mae, line_mae = row["final_test_mae"], row["baseline_holdout_mae"]
        axes[0].plot([position, position], [model_mae, line_mae], color=GRID,
                     linewidth=3, solid_capstyle="round", zorder=1)
        axes[0].plot([position], [line_mae], marker="o", markersize=9, color=LINE_REF,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        axes[0].plot([position], [model_mae], marker="o", markersize=9,
                     color=colour_of(row), markeredgecolor=SURFACE,
                     markeredgewidth=1.5, zorder=4)
        # Same sign convention as the right panel, so both read the same way.
        axes[0].annotate(f"{line_mae - model_mae:+.3f}", (position, min(model_mae, line_mae)),
                         textcoords="offset points", xytext=(0, -15), ha="center",
                         fontsize=7.5, color=INK_2)
    axes[0].set_xticks(positions, view["label"], rotation=20, ha="right")
    axes[0].set_ylabel("Holdout MAE — lower is better")
    axes[0].set_title("Model versus the line, on the same games")
    axes[0].set_xlim(-0.6, len(view) - 0.4)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=9,
               color=colour_of(view.iloc[0]), label="model (strategy colour)"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=LINE_REF,
               label="bookmaker line"),
    ], fontsize=8)

    colours = [GOOD if value > 0 else CRITICAL for value in view["holdout_edge"]]
    bars = axes[1].bar(positions, view["holdout_edge"], 0.55, color=colours)
    axes[1].axhline(0, color=AXIS, linewidth=1.2)
    axes[1].set_xticks(positions, view["label"], rotation=20, ha="right")
    axes[1].set_ylabel("Line MAE − model MAE")
    axes[1].set_title("MAE edge over the line\n(above zero = the model forecasts better)")
    for bar, value in zip(bars, view["holdout_edge"], strict=True):
        axes[1].annotate(f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2, value),
                         textcoords="offset points", xytext=(0, 3 if value >= 0 else -11),
                         ha="center", fontsize=8, color=INK_2)

    plt.tight_layout()
    plt.show()
    return view


# --------------------------------------------------------------- classifiers
def plot_classifier_calibration(runs: pd.DataFrame) -> pd.DataFrame | None:
    classifiers = runs[runs["is_classifier"]]
    if classifiers.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1.2, zorder=1)
    plotted = 0
    for _, row in classifiers.iterrows():
        for source, buckets in loaders.load_calibration_buckets(row):
            axes[0].plot(buckets["mean_predicted"], buckets["observed_frequency"],
                         marker="o", linestyle="-" if source == "CV" else "--",
                         color=colour_of(row), markeredgecolor=SURFACE,
                         markeredgewidth=1.2, label=f"{row['label']} · {source}", zorder=3)
            # Bucket population, so a point that is wildly off but backed by six
            # games is not read as a calibration failure.
            for _, bucket in buckets.iterrows():
                axes[0].annotate(f"n={int(bucket['n'])}",
                                 (bucket["mean_predicted"], bucket["observed_frequency"]),
                                 textcoords="offset points", xytext=(6, -10),
                                 fontsize=7, color=MUTED)
            plotted += 1
    axes[0].set_xlabel("Stated probability of OVER")
    axes[0].set_ylabel("Observed frequency of OVER")
    axes[0].set_title("Reliability — on the diagonal means the probabilities are honest")
    if plotted:
        axes[0].legend(fontsize=7.5, loc="best")
    else:
        axes[0].text(0.5, 0.5, "no calibration buckets saved", ha="center", va="center",
                     transform=axes[0].transAxes, color=MUTED)

    positions = np.arange(len(classifiers))
    width = 0.38
    axes[1].bar(positions - width / 2, classifiers["cv_log_loss_improvement"], width,
                color=[colour_of(r) for _, r in classifiers.iterrows()], label="CV")
    axes[1].bar(positions + width / 2, classifiers["log_loss_improvement"], width,
                color=[colour_of(r) for _, r in classifiers.iterrows()], alpha=0.45,
                label="holdout")
    axes[1].axhline(0, color=AXIS, linewidth=1.2)
    axes[1].set_xticks(positions, classifiers["label"], rotation=15, ha="right")
    axes[1].set_ylabel("Log-loss improvement over base rate")
    axes[1].set_title("Does it beat always predicting the base rate?\n(above zero = yes)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    return classifiers


# --------------------------------------------------------------- confidence
def plot_confidence_calibration(
    runs: pd.DataFrame, *, n_buckets: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the model's own confidence predict winning?

    Confidence is not in the same units across strategies (points for
    regressors, expected value for the classifier), so the shared axis is each
    run's own confidence percentile. Returns ``(buckets, trend)``.
    """
    from scipy import stats as scipy_stats

    bucket_order = [
        f"{int(i * 100 / n_buckets)}–{int((i + 1) * 100 / n_buckets)}%"
        for i in range(n_buckets)
    ]
    bucket_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []

    for _, row in runs.iterrows():
        loaded = loaders.load_predictions(row)
        if loaded is None:
            continue
        frame, source = loaded
        if len(frame) < n_buckets * 10:
            continue
        frame = frame.assign(pct=frame["selection_score"].rank(pct=True))
        frame["bucket"] = pd.cut(frame["pct"], np.linspace(0, 1, n_buckets + 1),
                                 labels=bucket_order, include_lowest=True)
        grouped = frame.groupby("bucket", observed=True).agg(
            n=("won", "size"), win_rate=("won", "mean"))
        for bucket, stats in grouped.iterrows():
            bucket_rows.append({
                "label": row["label"], "prediction_strategy": row["prediction_strategy"],
                "bucket": str(bucket), "n": stats["n"], "win_rate": stats["win_rate"],
            })
        # ONE trend test rather than n_buckets per-bucket tests: with five
        # buckets you would expect roughly one spurious "significant" bucket.
        rho, p_value = scipy_stats.spearmanr(frame["selection_score"], frame["won"])
        trend_rows.append({
            "label": row["label"], "prediction_strategy": row["prediction_strategy"],
            "source": source, "n": len(frame), "rho": rho, "p_value": p_value,
        })

    buckets = pd.DataFrame(bucket_rows)
    trend = pd.DataFrame(trend_rows)
    if buckets.empty:
        return buckets, trend

    # Small multiples: a single panel would overlay six crossing lines with two
    # runs sharing each colour, making identity ambiguous.
    present = set(buckets["prediction_strategy"])
    strategies = [s for s in STRATEGY_COLOR if s in present]
    strategies += [s for s in buckets["prediction_strategy"].unique() if s not in strategies]
    fig, axes = plt.subplots(1, len(strategies), figsize=(5.4 * len(strategies), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    dashes = ["-", "--", ":", "-."]

    for ax, strategy in zip(axes, strategies, strict=True):
        colour = STRATEGY_COLOR.get(strategy, MUTED)
        subset = buckets[buckets["prediction_strategy"] == strategy]
        for index, (label, group) in enumerate(subset.groupby("label")):
            group = group.set_index("bucket").reindex(bucket_order).reset_index()
            ax.plot(group["bucket"], group["win_rate"], marker="o", color=colour,
                    linestyle=dashes[index % len(dashes)], markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=label, zorder=3)
        ax.axhline(BREAK_EVEN, color=CRITICAL, linestyle="--", linewidth=1.4, zorder=2)
        ax.set_title(short_name(strategy))
        ax.tick_params(axis="x", rotation=20)
        if subset["label"].nunique() >= 2:
            ax.legend(fontsize=7.5, loc="best")

    axes[0].set_ylabel("Win rate")
    axes[-1].annotate("break-even 52.38%", xy=(1, BREAK_EVEN),
                      xycoords=("axes fraction", "data"), textcoords="offset points",
                      xytext=(-4, 5), ha="right", fontsize=8, color=CRITICAL)
    fig.supxlabel("Confidence percentile within the run  (right = the model's strongest opinions)",
                  fontsize=9, color=INK_2)
    fig.suptitle("Do the bets the model is most sure about actually win more often?",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

    buckets.attrs["bucket_order"] = bucket_order
    return buckets, trend


# --------------------------------------------------------------- folds
def plot_fold_consistency(runs: pd.DataFrame) -> pd.DataFrame | None:
    frames = []
    for _, row in runs.iterrows():
        folds = loaders.load_fold_betting(row)
        if folds is None:
            continue
        folds = folds.assign(label=row["label"],
                             prediction_strategy=row["prediction_strategy"])
        frames.append(folds)
    if not frames:
        return None

    all_folds = pd.concat(frames, ignore_index=True)
    labels = all_folds["label"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5))

    for label in labels:
        group = all_folds[all_folds["label"] == label].sort_values("fold")
        axes[0].plot(group["fold"], group["roi"], marker="o",
                     color=STRATEGY_COLOR.get(group["prediction_strategy"].iloc[0], MUTED),
                     markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
    axes[0].axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("ROI")
    axes[0].set_title("ROI per cross-validation fold")
    axes[0].legend(fontsize=7.5, loc="best")

    data = [all_folds.loc[all_folds["label"] == label, "roi"].dropna() for label in labels]
    box = axes[1].boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True,
                          medianprops={"color": INK},
                          meanprops={"marker": "D", "markerfacecolor": INK,
                                     "markeredgecolor": INK, "markersize": 5})
    for patch, label in zip(box["boxes"], labels, strict=True):
        strategy = all_folds.loc[all_folds["label"] == label, "prediction_strategy"].iloc[0]
        colour = STRATEGY_COLOR.get(strategy, MUTED)
        patch.set_facecolor(colour)
        patch.set_alpha(0.45)
        patch.set_edgecolor(colour)
    axes[1].axhline(0, color=AXIS, linewidth=1.2)
    axes[1].set_ylabel("Fold ROI")
    axes[1].set_title("Spread of fold ROI\n(a box straddling zero = no stable edge)")
    rotate_xticks(axes[1])

    plt.tight_layout()
    plt.show()

    consistency = (
        all_folds.groupby("label")
        .agg(folds=("fold", "count"), profitable=("roi", lambda s: int((s > 0).sum())),
             mean_roi=("roi", "mean"), sd_roi=("roi", "std"),
             worst=("roi", "min"), best=("roi", "max"))
        .reset_index()
    )
    consistency["mean_over_sd"] = consistency["mean_roi"] / consistency["sd_roi"]
    return consistency


# --------------------------------------------------------------- walk-forward
def plot_walk_forward(runs: pd.DataFrame) -> pd.DataFrame | None:
    walks = []
    for _, row in runs.iterrows():
        walk = loaders.load_walk_forward(row)
        if walk is not None:
            walks.append({"row": row, **walk})
    if not walks:
        return None

    audit = pd.DataFrame([{
        "label": w["row"]["label"],
        "game_days": len(w["daily"]),
        "leakage_days": int((w["daily"]["train_end_date"] >= w["daily"]["date"]).sum()),
    } for w in walks])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5))
    for walk in walks:
        row, predictions = walk["row"], walk["predictions"]
        colour = colour_of(row)
        threshold = row["bet_min_edge"] if pd.notna(row["bet_min_edge"]) else 0.0
        score = (predictions["selection_score"] if "selection_score" in predictions
                 else predictions["predicted_edge"].abs())
        bets = predictions[score > threshold].copy()
        if not bets.empty:
            settled = loaders.settle_bets(bets)
            settled["profit"] = np.where(
                settled["won"] == 1, DECIMAL_ODDS - 1.0,
                np.where(settled["push"], 0.0, -1.0),
            )
            axes[0].plot(settled["date"], settled["profit"].cumsum(), color=colour,
                         label=f"{row['label']} ({len(settled)} bets)", zorder=3)
        axes[1].plot(walk["daily"]["date"], walk["daily"]["train_n_games"],
                     color=colour, label=row["label"], zorder=3)

    axes[0].axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    axes[0].set_title("Cumulative profit across the test period (1 unit per bet)")
    axes[0].set_ylabel("Units")
    axes[1].set_title("Training-set size per retrain\n(flat = rolling window, rising = expanding)")
    axes[1].set_ylabel("Games in training set")
    for ax in axes:
        ax.set_xlabel("Game date")
        rotate_xticks(ax, 25)
        ax.legend(fontsize=7.5, loc="best")
    plt.tight_layout()
    plt.show()
    return audit


# --------------------------------------------------------------- lines
def plot_line_comparison(runs: pd.DataFrame) -> pd.DataFrame | None:
    frames = []
    for _, row in runs.iterrows():
        frame = loaders.load_line_comparison(row)
        if frame is None:
            continue
        frame = frame.copy()
        frame.insert(0, "label", row["label"])
        frame["prediction_strategy"] = row["prediction_strategy"]
        # Whether the settled-into line is itself a model input decides whether
        # any alternative-line gain is real or circular.
        frame["line_is_a_feature"] = frame["line_col"].isin(loaders.load_feature_names(row))
        frames.append(frame)
    if not frames:
        return None

    lines = pd.concat(frames, ignore_index=True)
    labels = lines["label"].unique().tolist()
    line_names = lines["line_col"].unique().tolist()
    positions = np.arange(len(labels))
    width = 0.8 / max(len(line_names), 1)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for offset, line_name in enumerate(line_names):
        subset = lines[lines["line_col"] == line_name].set_index("label").reindex(labels)
        shift = (offset - (len(line_names) - 1) / 2) * width
        colours = [colour_of(runs[runs["label"] == label].iloc[0]) for label in labels]
        # The settled-into line keeps full strength; alternatives are lighter,
        # since they are counterfactuals rather than results.
        bars = ax.bar(positions + shift, subset["roi"], width * 0.9, color=colours,
                      alpha=1.0 if offset == 0 else 0.45,
                      label=line_name.replace("TOTAL_LINE_", ""))
        for bar, value in zip(bars, subset["roi"], strict=True):
            if pd.isna(value):
                continue
            ax.annotate(f"{value:+.1%}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3 if value >= 0 else -11),
                        ha="center", fontsize=7.5, color=INK_2)
    ax.axhline(0, color=AXIS, linewidth=1.2)
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel("ROI")
    ax.set_title("ROI by the line bets settle into — same model, same predictions")
    ax.legend(fontsize=8, title="line", title_fontsize=8)
    plt.tight_layout()
    plt.show()
    return lines
