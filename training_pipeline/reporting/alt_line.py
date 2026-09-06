"""Re-settling a closing-line model's holdout bets at an earlier snapshot line.

The question this answers: a model trained and scored against the CLOSING line
picked a side on each holdout game. If those same picks had been settled
against the line quoted six hours before tip-off instead, would they have won
more often?

**This is deliberately not a fair test, and the number it produces is not a
strategy.** Two separate leaks, both in the optimistic direction:

1. *The decision uses information from after the price.* The model's features
   are the closing-line dataset's, so the pick is made with knowledge the
   market only had at the close. Settling it at T-360 prices a decision that
   could not have been made at T-360.
2. *For ``line_error_regressor`` the target itself is anchored to the close.*
   Its prediction IS ``TOTAL - closing_line``, so recovering an implied total
   (``prediction + closing_line``) puts the closing line straight into the
   number that is then compared against the earlier line. The closing line is
   not merely a scoring choice there; it is half the prediction.

So read the result as an upper bound on what earlier prices could be worth, and
as a measure of how much of the closing line's information the model is
actually adding to. A genuine answer needs a model trained on the
intermediate-line dataset -- which is exactly what the pooled and T-360 runs in
this campaign are, and they are the ones to compare against.

What is honest here
-------------------
The line swap itself, and the cohort control. Both sides of the comparison are
scored on the SAME games -- only those with a T-360 line -- so a difference in
win rate is a difference in settlement line and not a difference in which games
were counted. Every bet still goes through ``betting.evaluate_betting``
unchanged; nothing is refitted, and no decision is re-taken.

Joining predictions back to their games
---------------------------------------
The saved prediction frames carry no ``GAME_ID``: they carry a positional index
into a cleaned frame this module cannot rebuild. They do carry the date, the
realised total and the settlement line, and those three together identify a
game in the source CSV. :func:`attach_game_ids` uses that triple, and it
**discards any key that is not unique** in the source rather than accepting the
first match -- an ambiguous join would attribute a prediction to the wrong game
and every number after it would be quietly wrong. The match rate is returned so
the caller can report it instead of assuming it was 100%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nba_ou.config.market_columns import Market
from nba_ou.config.odds_columns import spread_line_home_col, total_line_col

from training_pipeline.betting import evaluate_betting, outcome_from_predictions
from training_pipeline.config import SNAPSHOT_COLUMN, PredictionStrategy
from training_pipeline.reporting import coverage
from training_pipeline.reporting.loaders import settle_bets
from training_pipeline.reporting.theme import (
    BREAK_EVEN,
    DECIMAL_ODDS,
    GRID,
    INK,
    INK_2,
    MUTED,
)

#: Which raw column of the source CSV carries the realised outcome for each
#: market -- the join key below always uses this alongside date and line.
OUTCOME_COL_BY_MARKET: dict[Market, str] = {
    Market.TOTALS: "TOTAL_POINTS",
    Market.SPREAD: "HOME_MARGIN",
}

#: Colours for the two settlement lines. Not the strategy palette: the series
#: here are two prices for one model, not two models.
CLOSING_COLOR = "#244a6b"
ALT_COLOR = "#8a5a9b"


class AlternativeLineError(RuntimeError):
    """The predictions could not be re-settled against a trustworthy line."""


def target_line_column(
    config: dict[str, Any], *, market: Market = Market.TOTALS
) -> str:
    """The column a run's bets settled into, from its flattened config.

    ``line_col`` is set for the total-points regressor and the classifier. It
    is deliberately absent for the two closing-line residual regressors
    (``line_error_regressor`` and ``spread_error_regressor``), whose target is
    built by subtracting the main book's line for their own market, so that
    book's column for ``market`` is the answer there -- see
    ``data.prepare_dataset``.
    """
    line_col = config.get("line_col")
    if line_col:
        return str(line_col)
    return spread_line_home_col() if market is Market.SPREAD else total_line_col()


def attach_game_ids(
    predictions: pd.DataFrame,
    source_csv: str | Path,
    *,
    line_col: str,
    outcome_col: str = "TOTAL_POINTS",
    game_id_col: str = "GAME_ID",
    date_col: str = "GAME_DATE",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add ``game_id`` to a prediction frame, dropping ambiguous matches.

    Reads four columns from ``source_csv``, not the whole file: the closing
    dataset is ~1,900 columns wide and none of the others are needed.

    ``outcome_col`` is which realised-outcome column of ``source_csv`` to join
    on -- ``TOTAL_POINTS`` for the totals market, ``HOME_MARGIN`` for the
    spread. The predictions side is read through ``outcome_from_predictions``,
    which finds the outcome whichever name the run wrote it under, so this
    works for both markets without the caller inspecting ``predictions``.

    Returns the matched rows and a report of what happened to the rest, so the
    caller can print the match rate rather than assume it.
    """
    source_csv = Path(source_csv)
    if not source_csv.exists():
        raise AlternativeLineError(
            f"Source dataset {source_csv} not found, so predictions cannot be "
            "joined back to their games."
        )
    source = pd.read_csv(
        source_csv,
        usecols=[game_id_col, date_col, outcome_col, line_col],
        dtype={game_id_col: str},
    )
    source[date_col] = pd.to_datetime(source[date_col])
    source = source.rename(
        columns={line_col: "target_line", outcome_col: "_join_outcome"}
    )

    # The three columns that together identify a game. Date alone matches ten
    # games; date plus the outcome still collides (two games a night can end
    # 231-all); adding the settlement line makes an undetected swap require
    # two games agreeing on all three at once. A key shared by two games
    # cannot identify either of them: dropping both sides loses a little
    # volume, but keeping one would silently mis-attribute.
    join_columns = (date_col, "_join_outcome", "target_line")
    ambiguous = source.duplicated(list(join_columns), keep=False)
    usable = source.loc[~ambiguous, [game_id_col, *join_columns]]

    frame = predictions.copy()
    date_source = "date" if "date" in frame.columns else date_col
    frame[date_col] = pd.to_datetime(frame[date_source])
    frame["_join_outcome"] = outcome_from_predictions(frame)

    merged = frame.merge(
        usable.rename(columns={game_id_col: "game_id"}),
        on=list(join_columns),
        how="left",
        validate="many_to_one",
    )
    matched = (
        merged[merged["game_id"].notna()]
        .drop(columns=["_join_outcome"])
        .reset_index(drop=True)
    )
    return matched, {
        "n_predictions": int(len(predictions)),
        "n_matched": int(len(matched)),
        "n_unmatched": int(len(predictions) - len(matched)),
        "n_ambiguous_keys_in_source": int(ambiguous.sum()),
    }


def snapshot_line_lookup(
    snapshot_csv: str | Path,
    *,
    snapshot_minutes: int,
    line_col: str,
    game_id_col: str = "GAME_ID",
    snapshot_col: str = SNAPSHOT_COLUMN,
) -> pd.Series:
    """``game_id -> line`` at one pre-game horizon, from the snapshot dataset.

    Within a single ``snapshot_col`` value the intermediate dataset holds one
    row per game, so this is a genuine lookup rather than an aggregate. A game
    missing that horizon simply does not appear.
    """
    snapshot_csv = Path(snapshot_csv)
    if not snapshot_csv.exists():
        raise AlternativeLineError(
            f"Snapshot dataset {snapshot_csv} not found, so no alternative "
            "line is available."
        )
    frame = pd.read_csv(
        snapshot_csv,
        usecols=[game_id_col, snapshot_col, line_col],
        dtype={game_id_col: str},
    )
    at_horizon = frame[frame[snapshot_col] == snapshot_minutes]
    if at_horizon.empty:
        raise AlternativeLineError(
            f"No rows at {snapshot_col}={snapshot_minutes} in {snapshot_csv.name}. "
            f"Available: {sorted(frame[snapshot_col].dropna().unique())}."
        )
    if at_horizon[game_id_col].duplicated().any():
        raise AlternativeLineError(
            f"{snapshot_csv.name} holds more than one row per game at "
            f"{snapshot_col}={snapshot_minutes}; the lookup would be ambiguous."
        )
    return (
        at_horizon.set_index(game_id_col)[line_col]
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
    )


def swap_settlement_line(frame: pd.DataFrame, alt_line: pd.Series) -> pd.DataFrame:
    """Re-settle each prediction against ``alt_line``, keyed by ``game_id``.

    The prediction is first put back into outcome space -- TOTAL_POINTS for the
    totals market, HOME_MARGIN for the spread. ``predicted_edge + target_line``
    recovers the implied outcome for BOTH regressors of either market: a
    total-points model's edge is ``prediction - line``, and a line-error (or
    spread-error) model's edge is the prediction itself, measured from its own
    line. That is the same translation ``line_scoring.predicted_total_points``
    performs, kept in one expression here because the artifacts carry the edge
    rather than the raw target.

    The side of the bet is then re-taken against the new line, because that is
    the whole mechanism: a game the model liked by half a point against the
    close can be an UNDER against a line that has since moved up. Rows without
    an alternative line are dropped.
    """
    if "game_id" not in frame.columns:
        raise AlternativeLineError(
            "swap_settlement_line needs a game_id column; call attach_game_ids first."
        )
    swapped = frame.copy()
    swapped["closing_line"] = swapped["target_line"]
    swapped["implied_total"] = swapped["predicted_edge"] + swapped["target_line"]
    swapped["target_line"] = swapped["game_id"].map(alt_line)
    swapped = swapped[swapped["target_line"].notna()].copy()

    swapped["predicted_edge"] = swapped["implied_total"] - swapped["target_line"]
    # Regressor selection is |edge| by definition. A classifier's score is a
    # maximum of two expected values and needs probabilities and prices that
    # the artifacts do not carry, which is why callers must exclude them.
    swapped["selection_score"] = swapped["predicted_edge"].abs()
    swapped["line_move"] = swapped["target_line"] - swapped["closing_line"]
    return settle_bets(swapped).reset_index(drop=True)


def compare_settlement_lines(
    closing: pd.DataFrame,
    swapped: pd.DataFrame,
    *,
    label: str = "",
    alt_name: str = "T-360",
    coverage_grid: tuple[float, ...] = coverage.COVERAGE_GRID,
    flat_decimal_odds: float = DECIMAL_ODDS,
) -> pd.DataFrame:
    """Both settlement lines over the same games, at every coverage level.

    ``closing`` is restricted to the games ``swapped`` retained before either
    is scored. Without that, the comparison mixes a change of settlement line
    with a change of cohort, and the cohort change alone can move a win rate by
    more than the effect being measured.
    """
    shared = set(swapped["game_id"])
    cohort = closing[closing["game_id"].isin(shared)].reset_index(drop=True)

    def selected_at(frame: pd.DataFrame, target: float) -> tuple[float, set[Any]]:
        cutoff = coverage.cutoff_for_coverage(frame["selection_score"], target)
        keep = frame.loc[frame["selection_score"] > cutoff, "game_id"]
        return cutoff, set(keep)

    # How much of the two series' selections actually coincide, per coverage.
    # The COHORT is identical by construction above, but the SELECTION is not:
    # each side ranks by its own margin, and swapping the settlement line
    # re-ranks the games. At 100% that is vacuous and the two series really are
    # the same games; by 60% they share about half their picks, so part of any
    # gap below full coverage is a different subset rather than a different
    # price. Reported rather than hidden, because the chart used to assert the
    # opposite.
    overlap: dict[float, dict[str, float]] = {}
    for target in coverage_grid:
        _, left = selected_at(cohort, target)
        _, right = selected_at(swapped, target)
        union = left | right
        overlap[target] = {
            "n_selected_shared": float(len(left & right)),
            "share_selection_shared": (
                float(len(left & right) / len(union)) if union else float("nan")
            ),
        }

    rows: list[dict[str, Any]] = []
    for name, frame in (("closing line", cohort), (f"{alt_name} line", swapped)):
        for target in coverage_grid:
            cutoff, _ = selected_at(frame, target)
            metrics = evaluate_betting(
                predicted_edge=frame["predicted_edge"],
                actual_total=outcome_from_predictions(frame),
                line=frame["target_line"],
                selection_score=frame["selection_score"],
                min_edge=cutoff,
                flat_decimal_odds=flat_decimal_odds,
            ).model_dump()
            row = {
                "label": label,
                "settled_at": name,
                "target_coverage": target,
                "cutoff": cutoff,
                **metrics,
                **overlap[target],
            }
            row["realised_coverage"] = row.pop("bet_rate")
            rows.append(row)
    return pd.DataFrame(rows)


def side_flip_summary(swapped: pd.DataFrame) -> dict[str, Any]:
    """How far the line moved, and how often that changed the pick.

    A win-rate difference with no side flips and no line movement would mean
    something is wrong with the join, so these are the numbers that make the
    comparison believable -- or not.
    """
    move = pd.to_numeric(swapped["line_move"], errors="coerce")
    closing_side = np.sign(swapped["implied_total"] - swapped["closing_line"])
    alt_side = np.sign(swapped["predicted_edge"])
    flipped = (closing_side != alt_side) & (closing_side != 0) & (alt_side != 0)
    return {
        "n_games": int(len(swapped)),
        "mean_abs_line_move": float(move.abs().mean()),
        "share_line_moved": float((move != 0).mean()),
        "n_side_flips": int(flipped.sum()),
        "share_side_flipped": float(flipped.mean()),
    }


def plot_settlement_comparison(
    comparison: pd.DataFrame,
    *,
    title: str = "",
    ax: Any = None,
    ylim: tuple[float, float] = coverage.WIN_RATE_YLIM,
) -> Any:
    """Win rate by coverage, one line per settlement price, same games.

    Paired lines rather than two panels: the whole point is the vertical gap
    between them at each coverage, and a gap is read far more accurately within
    one axis than across two.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)

    styles = {
        "closing line": {"color": CLOSING_COLOR, "marker": "o", "linestyle": "-"},
        None: {"color": ALT_COLOR, "marker": "^", "linestyle": "--"},
    }
    for name, group in comparison.groupby("settled_at", sort=False):
        group = group.sort_values("target_coverage", ascending=False)
        style = styles.get(str(name), styles[None])
        ax.plot(
            group["target_coverage"], group["win_rate"], label=str(name), **style
        )

    ax.axhline(BREAK_EVEN, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"break-even {BREAK_EVEN:.1%}", xy=(0.99, BREAK_EVEN),
        xycoords=("axes fraction", "data"), xytext=(0, 4),
        textcoords="offset points", ha="right", fontsize=8, color=INK_2,
    )
    ax.set_ylim(*coverage.win_rate_limits(comparison["win_rate"], ylim))
    ax.invert_xaxis()
    ax.set_xticks(sorted(comparison["target_coverage"].unique()))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of games kept, ranked by margin")
    ax.set_ylabel("win rate")
    ax.set_title(title, loc="left", color=INK, fontsize=9)
    ax.legend(fontsize=8, loc="best")
    # Not "same games on both series". The cohort is identical, the SELECTION
    # is not: each series keeps its own top share by its own margin, and the
    # line swap re-ranks the games. Only the 100% point compares like with
    # like, so say what the overlap actually is.
    note = "same cohort; only the 100% point selects the same games"
    if "share_selection_shared" in comparison.columns:
        shares = comparison.groupby("target_coverage")["share_selection_shared"].first()
        thinnest = shares.index.min()
        if pd.notna(shares.get(thinnest)):
            note += f" (at {thinnest:.0%} they share {shares[thinnest]:.0%} of picks)"
    ax.annotate(
        note, xy=(0.01, 0.02), xycoords="axes fraction",
        fontsize=7.5, color=MUTED,
    )
    return ax


def plot_settlement_summary(
    comparisons: pd.DataFrame,
    *,
    alt_name: str = "T-360",
    coverage_level: float = 1.0,
    label_map: Any = None,
    ax: Any = None,
) -> Any:
    """One row per run: holdout win rate at the close versus at ``alt_name``.

    The whole of section 6 in one picture, at ONE coverage -- 1.0 by default,
    every game. That restriction is what makes it simple and what makes it
    honest: at full coverage both series really are the same games, so the
    horizontal distance is the settlement line and nothing else. The
    coverage-swept chart answers a different, murkier question and is kept
    separate for that reason.

    A dumbbell rather than paired bars: the two values sit ~1-2 points apart on
    a base of ~52%, which grouped bars render as two identical rectangles.
    """
    view = comparisons[np.isclose(comparisons["target_coverage"], coverage_level)]
    if view.empty:
        raise ValueError(
            f"No rows at coverage {coverage_level:.0%}; the frame holds "
            f"{sorted(comparisons['target_coverage'].unique())}."
        )
    alt_column = f"{alt_name} line"
    wide = view.pivot_table(
        index="label", columns="settled_at", values=["win_rate", "n_bets"]
    )
    wide = wide.sort_values(("win_rate", "closing line"))
    names = [
        str(label_map.get(label, label)) if label_map is not None else str(label)
        for label in wide.index
    ]

    if ax is None:
        _, ax = plt.subplots(
            figsize=(9.0, 0.42 * len(wide) + 2.0), constrained_layout=True
        )
    positions = np.arange(len(wide))
    closing = wide[("win_rate", "closing line")].to_numpy()
    alternative = wide[("win_rate", alt_column)].to_numpy()

    ax.hlines(positions, closing, alternative, color=GRID, linewidth=3.0, zorder=1)
    ax.scatter(closing, positions, color=CLOSING_COLOR, s=54, zorder=3,
               label="settled at the close")
    ax.scatter(alternative, positions, color=ALT_COLOR, s=54, marker="^",
               zorder=3, label=f"settled at {alt_name}")
    ax.axvline(BREAK_EVEN, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"break-even {BREAK_EVEN:.1%}", xy=(BREAK_EVEN, 1.0),
        xycoords=("data", "axes fraction"), xytext=(4, -10),
        textcoords="offset points", fontsize=8, color=INK_2,
    )

    for y, (left, right) in enumerate(zip(closing, alternative, strict=True)):
        ax.annotate(
            f"{right - left:+.1%}", xy=(max(left, right), y), xytext=(9, 0),
            textcoords="offset points", va="center", fontsize=8, color=INK_2,
        )

    ax.set_yticks(positions, names, fontsize=9)
    ax.set_ylim(-0.7, len(wide) - 0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("holdout win rate — every game, no bet threshold")
    # Headroom on the right for the delta labels, which sit outside the marker.
    span = float(np.nanmax(np.r_[closing, alternative]) - np.nanmin(np.r_[closing, alternative]))
    ax.set_xlim(
        float(np.nanmin(np.r_[closing, alternative])) - 0.1 * max(span, 0.01),
        float(np.nanmax(np.r_[closing, alternative])) + 0.35 * max(span, 0.01),
    )
    n_games = int(view[view["settled_at"] == alt_column]["n_bets"].max())
    ax.set_title(
        "Same picks, two settlement lines\n"
        f"{coverage_level:.0%} of games · up to {n_games} per run",
        loc="left", color=INK, fontsize=10, fontweight="bold",
    )
    # Lower right: rows are sorted by closing win rate, so the bottom row is
    # the leftmost dumbbell and that corner stays clear.
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    return ax


class SettlementReport(NamedTuple):
    """Everything section 6 displays, built in one pass over the runs."""

    #: Per run and settlement line, win rate and ROI at every coverage level.
    comparisons: pd.DataFrame
    #: Per run, how the join went and how far the line moved.
    joins: pd.DataFrame


def closing_line_runs(runs: pd.DataFrame) -> list[tuple[Any, dict[str, Any]]]:
    """The runs this counterfactual applies to, with their flattened configs.

    Closing-line regressors only. An intermediate-line run is already scored at
    its own horizon, so re-settling it answers nothing; and a classifier's
    selection score is a maximum of two expected values, which needs
    probabilities and prices the prediction artifacts do not carry.
    """
    from training_pipeline.reporting.loaders import load_config_flat

    selected = []
    for _, run in runs.iterrows():
        config = load_config_flat(run)
        if config.get("data.dataset_type") == "closing_line" and not run["is_classifier"]:
            selected.append((run, config))
    return selected


def _run_market(run: Any) -> Market:
    """The betting market a run belongs to, from its ``prediction_strategy``."""
    return PredictionStrategy(str(run["prediction_strategy"])).market


def settlement_report(
    runs: pd.DataFrame,
    prediction_cache: dict[str, dict[str, pd.DataFrame]],
    *,
    project_root: Path,
    snapshot_csv: str | Path,
    snapshot_minutes: int,
    snapshot_book: str = "bet365",
    coverage_grid: tuple[float, ...] = coverage.COVERAGE_GRID,
) -> SettlementReport:
    """Re-settle every closing-line run's holdout at one earlier horizon.

    Handles both markets from the one snapshot dataset: a totals run is
    re-settled against ``ODDS_TOTAL_LINE_<snapshot_book>`` and a spread run
    against ``ODDS_SPREAD_LINE_HOME_<snapshot_book>``. Each market's snapshot
    lookup is read once, lazily, and only for the markets actually present
    among ``runs`` -- not once per run: the intermediate dataset is 2,500
    columns wide.
    """
    alt_name = f"T-{snapshot_minutes}"
    alt_lines_by_market: dict[Market, pd.Series] = {}

    def alt_lines_for(market: Market) -> pd.Series:
        if market not in alt_lines_by_market:
            line_col = (
                spread_line_home_col(snapshot_book) if market is Market.SPREAD
                else total_line_col(snapshot_book)
            )
            alt_lines_by_market[market] = snapshot_line_lookup(
                Path(project_root) / snapshot_csv,
                snapshot_minutes=snapshot_minutes,
                line_col=line_col,
            )
        return alt_lines_by_market[market]

    comparisons, joins = [], []
    for run, config in closing_line_runs(runs):
        holdout = prediction_cache.get(str(run["label"]), {}).get("holdout")
        if holdout is None or holdout.empty:
            continue
        market = _run_market(run)
        matched, report = attach_game_ids(
            holdout,
            Path(project_root) / str(config["data.csv_path"]),
            line_col=target_line_column(config, market=market),
            outcome_col=OUTCOME_COL_BY_MARKET[market],
        )
        swapped = swap_settlement_line(matched, alt_lines_for(market))
        joins.append({
            "label": run["label"], **report,
            f"n_with_{alt_name}_line": len(swapped),
            **side_flip_summary(swapped),
        })
        comparisons.append(
            compare_settlement_lines(
                matched, swapped, label=run["label"], alt_name=alt_name,
                coverage_grid=coverage_grid,
            )
        )

    return SettlementReport(
        comparisons=(
            pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
        ),
        joins=pd.DataFrame(joins),
    )


def settlement_gap(comparisons: pd.DataFrame, *, alt_name: str) -> pd.DataFrame:
    """The two settlement lines side by side, with the difference between them.

    The ``gain`` columns are ``alt - closing``, so a positive number means the
    earlier price paid better on the same games.
    """
    if comparisons.empty:
        return pd.DataFrame()
    alt_column = f"{alt_name} line"
    gap = comparisons.pivot_table(
        index=["label", "target_coverage"], columns="settled_at",
        values=["win_rate", "roi", "n_bets"],
    ).reset_index()
    for metric in ("win_rate", "roi"):
        gap[(metric, "gain")] = gap[(metric, alt_column)] - gap[(metric, "closing line")]
    return gap.sort_values(["label", "target_coverage"], ascending=[True, False])
