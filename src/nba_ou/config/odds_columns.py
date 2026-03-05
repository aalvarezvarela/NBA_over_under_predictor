from __future__ import annotations

import re

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


def total_line_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"TOTAL_LINE_{b}"


def spread_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"SPREAD_{b}"


def moneyline_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"MONEYLINE_{b}"


def extract_total_line_books(df: pd.DataFrame) -> list[str]:
    """
    Infer sportsbook names from columns shaped as TOTAL_LINE_<book>.
    Returns books in deterministic sorted order.
    """
    books = set()
    for col in df.columns:
        m = re.match(r"^TOTAL_LINE_(.+)$", col)
        if m:
            books.add(m.group(1))
    return sorted(books)


def resolve_main_total_line_col(
    df: pd.DataFrame, book: str | None = None
) -> str | None:
    """
    Resolve the active total-line column.
    Prefer configured book; fallback to first available TOTAL_LINE_* column.
    """
    preferred = total_line_col(book)
    if preferred in df.columns:
        return preferred

    candidates = extract_total_line_books(df)
    if not candidates:
        return None
    return f"TOTAL_LINE_{candidates[0]}"
