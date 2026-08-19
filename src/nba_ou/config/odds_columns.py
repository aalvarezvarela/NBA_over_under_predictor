from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from nba_ou.config.settings import SETTINGS

DEFAULT_MAIN_BOOK = "consensus_opener"
BOOK_ALIASES = {
    "bet_365": "bet365",
}


def get_main_book() -> str:
    configured = getattr(SETTINGS, "main_sportsbook", None)
    if configured is None:
        return DEFAULT_MAIN_BOOK
    configured = str(configured).strip()
    if not configured:
        return DEFAULT_MAIN_BOOK
    return BOOK_ALIASES.get(configured, configured)


#: Marker prefix for every column derived from betting-odds data, unified across both training
#: pipelines so odds features can be selected as easily as leakage-safe ``_BEFORE`` columns.
ODDS_COLUMN_PREFIXES: tuple[str, ...] = ("ODDS_",)


def is_odds_column(column: str) -> bool:
    """True if ``column`` carries the unified odds-derived marker prefix."""
    return column.startswith(ODDS_COLUMN_PREFIXES)


#: Column-name shapes that are odds-derived *by construction*: every column
#: whose name starts with one of these comes from a bookmaker market, whichever
#: route it took through the pipeline. Uppercase entries are the canonical
#: post-merge names built by ``total_line_col`` and friends; lowercase entries
#: are the raw per-book market columns as they arrive from the odds databases
#: (``total_<book>_price_over``, ``spread_<book>_line_home``, ``ml_<book>_price_home``,
#: ``moneyline_pct_bets_home``, ...) plus every rolling ``_BEFORE`` variant built
#: on top of them.
#:
#: This list is what makes the ``ODDS_`` marker an enforceable invariant rather
#: than a convention that holds only for the paths someone remembered to update:
#: a new odds feature named in any of these shapes is caught even if it never
#: passes through the rename helpers.
#:
#: Deliberately NOT included: ``DIFF_FROM_`` and ``IS_OVER_``. Both are
#: odds-derived by any reasonable definition, but they are named after the
#: *target* relationship rather than the market and are not currently prefixed
#: anywhere in the pipeline. Bringing them in is a separate, deliberate rename --
#: not something this guard should force silently.
ODDS_SHAPED_PREFIXES: tuple[str, ...] = (
    "TOTAL_LINE_",
    "SPREAD_",
    "MONEYLINE_",
    "total_",
    "spread_",
    "ml_",
    "moneyline_",
)


def is_odds_shaped_column(column: str) -> bool:
    """True if ``column`` is odds-derived by its name shape, prefix or not."""
    return column.startswith(ODDS_SHAPED_PREFIXES)


def strip_odds_prefix(column: str) -> str:
    """Return ``column`` without its leading ``ODDS_`` marker, if it has one."""
    return column.removeprefix("ODDS_") if is_odds_column(column) else column


def find_unprefixed_odds_columns(columns: Iterable[str]) -> list[str]:
    """Odds-derived columns that are missing the unified ``ODDS_`` marker."""
    return [
        column
        for column in columns
        if is_odds_shaped_column(column) and not is_odds_column(column)
    ]


def apply_odds_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix every odds-shaped column with ``ODDS_``.

    Idempotent: columns that already carry the marker are left alone, so this
    can be applied at the end of any pipeline without tracking whether an
    earlier stage already ran it.

    Must run **after** the last stage that reads raw market columns by their
    unprefixed names -- notably ``engineer_odds_features``, which resolves its
    inputs as ``total_<book>_price_over`` and would silently emit fewer features
    if the rename had already happened.
    """
    rename = {
        column: f"ODDS_{column}" for column in find_unprefixed_odds_columns(df.columns)
    }
    return df.rename(columns=rename) if rename else df


def assert_odds_columns_prefixed(columns: Iterable[str], *, context: str) -> None:
    """Fail if any odds-derived column reached the output without ``ODDS_``.

    The point of the marker is that ``[c for c in df.columns if is_odds_column(c)]``
    selects *every* odds feature. That only holds if nothing can slip through, so
    this raises rather than warns: an odds column arriving without the marker is
    invisible to every consumer that selects on it, and nothing else would ever
    surface the omission.
    """
    offenders = find_unprefixed_odds_columns(columns)
    if offenders:
        shown = ", ".join(offenders[:20])
        more = f" (+{len(offenders) - 20} more)" if len(offenders) > 20 else ""
        raise ValueError(
            f"{len(offenders)} odds-derived column(s) reached {context} without the "
            f"unified 'ODDS_' prefix: {shown}{more}. Every column named like a "
            "bookmaker market must carry the marker so it can be selected via "
            "nba_ou.config.odds_columns.is_odds_column(). Route the frame through "
            "apply_odds_prefix() or name the feature with the prefix at source."
        )


def total_line_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"ODDS_TOTAL_LINE_{b}"


def spread_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"ODDS_SPREAD_{b}"


def moneyline_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"ODDS_MONEYLINE_{b}"


def extract_total_line_books(df: pd.DataFrame) -> list[str]:
    """
    Infer sportsbook names from columns shaped as ODDS_TOTAL_LINE_<book>.
    Returns books in deterministic sorted order.
    """
    books = set()
    for col in df.columns:
        m = re.match(r"^ODDS_TOTAL_LINE_(.+)$", col)
        if m:
            books.add(m.group(1))
    return sorted(books)


def total_line_over_col_raw(book: str | None = None) -> str:
    """
    Get the raw odds data column name for total line over.
    Format: total_{book}_line_over (used in odds data before merge).
    """
    b = book or get_main_book()
    return f"ODDS_TOTAL_LINE_{b}"


def resolve_main_total_line_col(
    df: pd.DataFrame, book: str | None = None
) -> str | None:
    """
    Resolve the active total-line column.
    Prefer configured book; fallback to first available ODDS_TOTAL_LINE_* column.
    """
    preferred = total_line_col(book)
    if preferred in df.columns:
        return preferred

    candidates = extract_total_line_books(df)
    if not candidates:
        return None
    return f"ODDS_TOTAL_LINE_{candidates[0]}"
