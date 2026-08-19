"""Cross-book consensus, dispersion and steam, as of each snapshot.

Aggregates the per-book panel down to one row per (game, market, snapshot).
Everything is computed from books' states at T, so nothing here can see a price
that had not been posted.

The consensus is a **median**, not a mean: a single stale book sitting three
points off the market is common in this data, and a mean would drag the
consensus toward it. Dispersion is reported separately precisely so that
disagreement stays visible instead of being averaged away.

"Steam" here is the count and fraction of books that moved the same way inside
the last hour. Cross-book agreement over a short window is the classic
sharp-money signature, and unlike most such signals it is fully observable at T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONSENSUS_KEYS = ["game_id", "market", "snapshot_minutes"]


#: Window steam is measured over, when it is configured at all.
#:
#: Pinned rather than "the shortest available". Steam is *cross-book agreement*,
#: which needs enough books to have moved for agreement to mean anything, and
#: the short windows do not clear that bar: measured on the store a book moves
#: in the trailing 60 minutes on 32% of (row, book) and 37% of rows have no book
#: moving at all -- at 15 minutes steam would be a near-constant zero. Letting
#: the shortest configured window define it meant that merely *adding* a shorter
#: window to ``DEFAULT_WINDOWS`` silently redefined an existing feature family,
#: which is exactly what happened when 15 and 30 were introduced.
STEAM_WINDOW_MINUTES = 60


def steam_move_column(panel: pd.DataFrame) -> str:
    """The ``move_last_<w>`` column steam is measured over.

    Prefers ``STEAM_WINDOW_MINUTES`` and falls back to the shortest window on
    offer. The fallback is why this is derived at all rather than hardcoded: a
    caller passing ``windows=(120, 360)`` used to raise a bare ``KeyError`` deep
    inside the aggregation.
    """
    candidates = []
    for column in panel.columns:
        if column.startswith("move_last_"):
            suffix = column.removeprefix("move_last_")
            if suffix.isdigit():
                candidates.append((int(suffix), column))
    if not candidates:
        raise ValueError(
            "No move_last_<minutes> column found; run add_movement_features "
            "before aggregate_across_books."
        )
    preferred = f"move_last_{STEAM_WINDOW_MINUTES}"
    if any(column == preferred for _, column in candidates):
        return preferred
    return min(candidates)[1]


def _weighted_agreement(frame: pd.DataFrame, move_column: str) -> pd.Series:
    """Directional agreement across books over the shortest configured window.

    ``steam_fraction`` is the share of *quoting* books moving in the dominant
    direction: 0.0 when nobody moved, 0.5 on an even two-up/two-down split of
    four books, 1.0 when every quoting book agreed. It is deliberately not
    normalised by the number of movers, so "two books moved and agreed" scores
    lower than "five books moved and agreed" -- but that also means it cannot
    by itself distinguish a quiet market from a split one, which is why
    ``steam_movers`` is reported beside it.
    """
    directions = np.sign(frame[move_column].fillna(0.0))
    n_books = float(len(directions))
    n_up = float((directions > 0).sum())
    n_down = float((directions < 0).sum())
    dominant = max(n_up, n_down)
    movers = n_up + n_down

    return pd.Series(
        {
            "steam_books_up": n_up,
            "steam_books_down": n_down,
            "steam_net": n_up - n_down,
            "steam_movers": movers,
            "steam_fraction": (dominant / n_books) if n_books else np.nan,
            # Share of the books that actually moved which agreed: 1.0 when the
            # movers were unanimous, regardless of how many sat still.
            "steam_agreement": (dominant / movers) if movers else 0.0,
        }
    )


def aggregate_across_books(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, market, snapshot) summarising all books at T."""
    if panel.empty:
        return pd.DataFrame(columns=CONSENSUS_KEYS)

    move_column = steam_move_column(panel)
    grouped = panel.groupby(CONSENSUS_KEYS, sort=False)

    # Aggregated on ``level``, not ``norm_line``: the latter is NaN for the
    # whole moneyline market, which made every dispersion figure and the book
    # count itself zero there.
    consensus = grouped.agg(
        consensus_line=("level", "median"),
        consensus_line_mean=("level", "mean"),
        crossbook_std=("level", lambda s: s.std(ddof=0)),
        crossbook_range=("level", lambda s: s.max() - s.min()),
        consensus_norm_line=("norm_line", "median"),
        consensus_raw_line=("raw_line", "median"),
        consensus_fair_left=("fair_left", "median"),
        consensus_overround=("overround", "median"),
        n_books_quoting=("level", "count"),
        median_line_age=("line_age_minutes", "median"),
        max_line_age=("line_age_minutes", "max"),
        consensus_move_from_open=("move_from_open", "median"),
        consensus_move_recent=(move_column, "median"),
        consensus_n_moves=("n_moves_so_far", "median"),
        consensus_opener_line=("opener_line", "median"),
        consensus_has_quote=("has_quote", "sum"),
    )

    steam = grouped.apply(
        _weighted_agreement, move_column=move_column, include_groups=False
    )
    consensus = consensus.join(steam)

    consensus["crossbook_std"] = consensus["crossbook_std"].fillna(0.0)
    consensus["crossbook_range"] = consensus["crossbook_range"].fillna(0.0)
    return consensus.reset_index()


def add_book_deviation(panel: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Each book's distance from the consensus at the same instant.

    An outlying book is either the stale one or the sharp one, and the model is
    better placed than we are to decide which -- but it can only do that if the
    deviation is given to it explicitly.
    """
    if panel.empty:
        return panel

    merged = panel.merge(
        consensus[[*CONSENSUS_KEYS, "consensus_line", "crossbook_std"]],
        on=CONSENSUS_KEYS,
        how="left",
    )
    merged["deviation_from_consensus"] = merged["level"] - merged["consensus_line"]
    merged["abs_deviation_from_consensus"] = merged[
        "deviation_from_consensus"
    ].abs()
    # Scale-free version: "0.5 off" means something different on a tight market
    # than on a scattered one, and the raw gap is not comparable across markets
    # at all now that the moneyline is measured in probability.
    spread = merged["crossbook_std"].replace(0.0, np.nan)
    merged["deviation_z"] = (merged["deviation_from_consensus"] / spread).fillna(0.0)
    merged["is_outlier_book"] = (merged["deviation_z"].abs() > 1.5).astype(int)
    return merged.drop(columns=["crossbook_std"])
