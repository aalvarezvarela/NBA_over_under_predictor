"""Normalize asymmetrically priced NBA totals to an estimated 50/50 line.

The conversion assumes total points are normally distributed around the fair
market total.  A single alternate-line quote identifies its fair probability
after removing vig, while ``sigma`` controls the probability-to-points mapping.
"""

from math import floor, isclose
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_TOTAL_POINTS_SIGMA = 15.7
DEFAULT_QUOTE_INCREMENT = 0.5
DEFAULT_CENTERED_AMERICAN_ODDS = -110.0
DEFAULT_PRICE_DECIMAL_PLACES = 2

_STANDARD_NORMAL = NormalDist()
_TOTAL_LINE_OVER_SUFFIX = "_line_over"


def odds_to_decimal(odds: float, odds_format: str = "decimal") -> float:
    """Convert decimal or American odds to decimal odds."""
    normalized_format = odds_format.lower()
    value = float(odds)

    if normalized_format in {"decimal", "european", "eu"}:
        if not np.isfinite(value) or value <= 1.0:
            raise ValueError("Decimal odds must be finite and greater than 1.")
        return value

    if normalized_format in {"american", "us"}:
        if not np.isfinite(value) or value == 0.0:
            raise ValueError("American odds must be finite and non-zero.")
        if value > 0.0:
            return 1.0 + value / 100.0
        return 1.0 + 100.0 / abs(value)

    raise ValueError("odds_format must be 'decimal' or 'american'.")


def remove_vig_two_way(
    over_odds: float,
    under_odds: float,
    odds_format: str = "decimal",
) -> tuple[float, float, float]:
    """Return fair over/under probabilities and the raw bookmaker overround."""
    over_decimal = odds_to_decimal(over_odds, odds_format)
    under_decimal = odds_to_decimal(under_odds, odds_format)

    raw_over_probability = 1.0 / over_decimal
    raw_under_probability = 1.0 / under_decimal
    probability_sum = raw_over_probability + raw_under_probability

    return (
        raw_over_probability / probability_sum,
        raw_under_probability / probability_sum,
        probability_sum - 1.0,
    )


def round_to_increment(value: float, increment: float = 0.5) -> float:
    """Round a non-negative line half-up to a sportsbook quote increment."""
    if increment <= 0.0:
        raise ValueError("increment must be positive.")
    if value < 0.0 or not np.isfinite(value):
        raise ValueError("value must be finite and non-negative.")
    return floor(value / increment + 0.5) * increment


def _estimate_integer_line_center(
    line: float,
    fair_over_probability: float,
    sigma: float,
) -> float:
    """Estimate a center for an integer line, accounting for a possible push."""

    def conditional_over_probability(center: float) -> float:
        over_win_probability = 1.0 - _STANDARD_NORMAL.cdf(
            (line + 0.5 - center) / sigma
        )
        under_win_probability = _STANDARD_NORMAL.cdf(
            (line - 0.5 - center) / sigma
        )
        return over_win_probability / (
            over_win_probability + under_win_probability
        )

    low = line - 10.0 * sigma
    high = line + 10.0 * sigma
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if conditional_over_probability(midpoint) < fair_over_probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def estimate_centered_total_line(
    line: float,
    over_odds: float,
    under_odds: float,
    *,
    odds_format: str = "decimal",
    sigma: float = DEFAULT_TOTAL_POINTS_SIGMA,
    quote_increment: float = DEFAULT_QUOTE_INCREMENT,
) -> float:
    """Estimate and quote the fair 50/50 total from one two-way market quote."""
    line = float(line)
    if not np.isfinite(line) or line < 0.0:
        raise ValueError("line must be finite and non-negative.")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive.")

    fair_over_probability, _, _ = remove_vig_two_way(
        over_odds,
        under_odds,
        odds_format,
    )

    if isclose(line, round(line), abs_tol=1e-10):
        center = _estimate_integer_line_center(
            line,
            fair_over_probability,
            sigma,
        )
    else:
        center = line + sigma * _STANDARD_NORMAL.inv_cdf(
            fair_over_probability
        )

    return round_to_increment(center, quote_increment)


def _centered_price(odds_format: str, american_odds: float) -> float:
    normalized_format = odds_format.lower()
    if normalized_format in {"decimal", "european", "eu"}:
        return odds_to_decimal(american_odds, "american")
    if normalized_format in {"american", "us"}:
        if american_odds >= 0.0:
            raise ValueError("centered_american_odds must be negative.")
        return float(american_odds)
    raise ValueError("odds_format must be 'decimal' or 'american'.")


def _total_market_prefixes(columns: pd.Index) -> list[str]:
    """Find bookmaker prefixes that contain a complete totals quote quartet."""
    column_names = set(columns)
    prefixes: list[str] = []
    for column in columns:
        if not column.startswith("total_") or not column.endswith(
            _TOTAL_LINE_OVER_SUFFIX
        ):
            continue
        prefix = column[: -len(_TOTAL_LINE_OVER_SUFFIX)]
        required_columns = {
            f"{prefix}_line_over",
            f"{prefix}_line_under",
            f"{prefix}_price_over",
            f"{prefix}_price_under",
        }
        if required_columns.issubset(column_names):
            prefixes.append(prefix)
    return prefixes


def normalize_total_lines_inplace(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    odds_format: str = "decimal",
    sigma: float = DEFAULT_TOTAL_POINTS_SIGMA,
    quote_increment: float = DEFAULT_QUOTE_INCREMENT,
    centered_american_odds: float = DEFAULT_CENTERED_AMERICAN_ODDS,
    price_decimal_places: int = DEFAULT_PRICE_DECIMAL_PLACES,
) -> pd.DataFrame:
    """Center every complete, asymmetrically priced bookmaker total in ``df``.

    The existing over line, under line, over price, and under price columns are
    modified directly. Prices are first rounded to ``price_decimal_places`` so
    insignificant differences are ignored. No columns are added. A quote is
    normalized only when all four values are valid, both sides refer to the
    same total, their rounded prices differ, and neither side is already at the
    standard centered price (``-110`` by default). The returned object is the
    same DataFrame instance.
    """
    if not enabled or df.empty:
        return df
    if not isinstance(price_decimal_places, int) or price_decimal_places < 0:
        raise ValueError("price_decimal_places must be a non-negative integer.")

    centered_price = round(
        _centered_price(odds_format, centered_american_odds),
        price_decimal_places,
    )
    total_comparable_markets = 0
    total_changed_markets = 0

    for prefix in _total_market_prefixes(df.columns):
        line_over_col = f"{prefix}_line_over"
        line_under_col = f"{prefix}_line_under"
        price_over_col = f"{prefix}_price_over"
        price_under_col = f"{prefix}_price_under"

        line_over = pd.to_numeric(df[line_over_col], errors="coerce")
        line_under = pd.to_numeric(df[line_under_col], errors="coerce")
        price_over = pd.to_numeric(df[price_over_col], errors="coerce").round(
            price_decimal_places
        )
        price_under = pd.to_numeric(df[price_under_col], errors="coerce").round(
            price_decimal_places
        )

        valid_over_prices = price_over.notna()
        valid_under_prices = price_under.notna()
        df.loc[valid_over_prices, price_over_col] = price_over.loc[valid_over_prices]
        df.loc[valid_under_prices, price_under_col] = price_under.loc[
            valid_under_prices
        ]

        comparable_mask = (
            line_over.notna()
            & line_under.notna()
            & price_over.notna()
            & price_under.notna()
            & np.isclose(line_over, line_under, rtol=0.0, atol=1e-10)
        )
        total_comparable_markets += int(np.count_nonzero(comparable_mask))
        has_centered_side = np.isclose(
            price_over,
            centered_price,
            rtol=0.0,
            atol=1e-10,
        ) | np.isclose(
            price_under,
            centered_price,
            rtol=0.0,
            atol=1e-10,
        )
        candidate_mask = (
            comparable_mask
            & ~has_centered_side
            & ~np.isclose(
                price_over,
                price_under,
                rtol=0.0,
                atol=1e-10,
            )
        )

        column_positions = {
            column: df.columns.get_loc(column)
            for column in (
                line_over_col,
                line_under_col,
                price_over_col,
                price_under_col,
            )
        }
        candidate_positions = np.flatnonzero(np.asarray(candidate_mask))
        for row_position in candidate_positions:
            try:
                centered_line = estimate_centered_total_line(
                    line_over.iloc[row_position],
                    price_over.iloc[row_position],
                    price_under.iloc[row_position],
                    odds_format=odds_format,
                    sigma=sigma,
                    quote_increment=quote_increment,
                )
            except (TypeError, ValueError, ZeroDivisionError):
                continue

            df.iat[row_position, column_positions[line_over_col]] = centered_line
            df.iat[row_position, column_positions[line_under_col]] = centered_line
            df.iat[row_position, column_positions[price_over_col]] = centered_price
            df.iat[row_position, column_positions[price_under_col]] = centered_price
            total_changed_markets += 1

    changed_percentage = (
        100.0 * total_changed_markets / total_comparable_markets
        if total_comparable_markets
        else 0.0
    )
    print(
        "Total-line normalization: "
        f"{total_changed_markets}/{total_comparable_markets} bookmaker-game "
        f"markets changed ({changed_percentage:.2f}%)."
    )

    return df
