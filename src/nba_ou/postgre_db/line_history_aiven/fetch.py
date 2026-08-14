"""Read pre-game ticks out of the Aiven line-history store.

Read-only. The store's storage encodings are undone exactly once -- here -- so
that no feature module downstream has to remember them:

* ``left_line``/``right_line`` are half-points doubled into ``SMALLINT``
  (``449 -> 224.5``). Divided by 2 on the way out.
* ``left_price``/``right_price`` are **American** odds, already null where the
  book was off the board.
* ``mins_to_tip`` is negative pre-game. Callers think in *positive* minutes
  before tip, so this module emits ``minutes_before_tip = -mins_to_tip`` and the
  sign convention never leaks into feature code.

Per-market column semantics (see ``transform.py`` and the Phase 0 findings):

* ``totals`` -- ``left`` is OVER, ``right`` is UNDER, and a valid quote has
  ``left_line == right_line``.
* ``point_spread`` -- mirrored, ``left_line == -right_line``. The price-bleed
  repair leaves some rows with a valid price and a NULL line, so a present price
  does not imply a present line.
* ``money_line`` -- prices only; both line columns are NULL by nature.
"""

from __future__ import annotations

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.postgre_db.config.db_config import connect_line_history_db

from .schema import SCHEMA

#: Market codes as stored in ``lh_market``.
MARKET_TOTALS = "totals"
MARKET_SPREAD = "point_spread"
MARKET_MONEYLINE = "money_line"
ALL_MARKETS: tuple[str, ...] = (MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE)

#: Books that only ever appear in part of the history. Carrying one of these
#: unflagged lets a model recover the season from mere column availability.
PARTIAL_COVERAGE_BOOKS: tuple[str, ...] = ("fanatics_sportsbook",)

#: Safety margin on the leakage filter. Phase 0 recommends staying clear of the
#: tipoff boundary rather than trusting it to the minute, because a per-game
#: tipoff that is a few minutes stale would otherwise admit an in-play tick.
DEFAULT_MIN_MINUTES_BEFORE_TIP = 5

TICK_COLUMNS = [
    "game_id",
    "season_year",
    "market",
    "book",
    "line_ts",
    "minutes_before_tip",
    "is_opener",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]

GAME_COLUMNS = [
    "game_id",
    "game_date",
    "season_year",
    "tipoff_utc",
    "team_home",
    "team_away",
]


def _rows_to_frame(cursor: psycopg.Cursor, columns: list[str]) -> pd.DataFrame:
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _decode_line(values: pd.Series) -> pd.Series:
    """Undo the doubled-half-point SMALLINT encoding."""
    return pd.to_numeric(values, errors="coerce") / 2.0


def fetch_games(
    season_years: list[int],
    *,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """Return the ``lh_game`` dimension for ``season_years``."""
    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        query = sql.SQL(
            """
            SELECT game_id, game_date, season_year, tipoff_utc,
                   team_home, team_away
            FROM {}.lh_game
            WHERE season_year = ANY(%s)
            ORDER BY game_date, game_id
            """
        ).format(sql.Identifier(SCHEMA))
        with conn.cursor() as cur:
            cur.execute(query, ([int(s) for s in season_years],))
            games = _rows_to_frame(cur, GAME_COLUMNS)
    finally:
        if owned:
            conn.close()

    if games.empty:
        return games

    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["tipoff_utc"] = pd.to_datetime(games["tipoff_utc"], utc=True)
    games["season_year"] = pd.to_numeric(games["season_year"]).astype("int64")
    return games.reset_index(drop=True)


def fetch_pregame_ticks(
    season_years: list[int],
    *,
    markets: tuple[str, ...] = ALL_MARKETS,
    exclude_books: tuple[str, ...] = (),
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """Return every pre-game tick for ``season_years``, decoded.

    The ``is_pregame`` flag alone is not the leakage filter -- it only separates
    pre-game from in-play. Snapshot horizons are applied later, per snapshot, in
    ``snapshots.py``; this function's ``min_minutes_before_tip`` is only the
    blanket safety margin around the tipoff boundary itself.
    """
    if not markets:
        raise ValueError("markets must not be empty.")
    unknown = set(markets) - set(ALL_MARKETS)
    if unknown:
        raise ValueError(f"Unknown market(s): {sorted(unknown)}")
    if min_minutes_before_tip < 0:
        raise ValueError("min_minutes_before_tip must be >= 0.")

    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        query = sql.SQL(
            """
            SELECT l.game_id, l.season_year, m.code, b.slug,
                   l.line_ts, l.mins_to_tip, l.is_opener,
                   l.left_line, l.left_price, l.right_line, l.right_price
            FROM {schema}.lh_line l
            JOIN {schema}.lh_market m USING (market_id)
            JOIN {schema}.lh_book   b USING (book_id)
            WHERE l.season_year = ANY(%s)
              AND l.is_pregame
              AND l.mins_to_tip <= %s
              AND m.code = ANY(%s)
              AND NOT (b.slug = ANY(%s))
            ORDER BY l.game_id, m.code, b.slug, l.line_ts
            """
        ).format(schema=sql.Identifier(SCHEMA))
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    [int(s) for s in season_years],
                    -int(min_minutes_before_tip),
                    list(markets),
                    list(exclude_books),
                ),
            )
            ticks = _rows_to_frame(cur, TICK_COLUMNS)
    finally:
        if owned:
            conn.close()

    if ticks.empty:
        return ticks

    ticks["game_id"] = ticks["game_id"].astype(str)
    ticks["line_ts"] = pd.to_datetime(ticks["line_ts"], utc=True)
    ticks["season_year"] = pd.to_numeric(ticks["season_year"]).astype("int64")

    # Stored as negative minutes relative to tip; downstream thinks in positive
    # "minutes before tip", which is also how the snapshot grid is expressed.
    ticks["minutes_before_tip"] = -pd.to_numeric(
        ticks["minutes_before_tip"], errors="coerce"
    ).astype("float64")

    ticks["is_opener"] = ticks["is_opener"].astype(bool)

    for column in ["left_line", "right_line"]:
        ticks[column] = _decode_line(ticks[column])
    for column in ["left_price", "right_price"]:
        ticks[column] = pd.to_numeric(ticks[column], errors="coerce")

    return ticks.reset_index(drop=True)


def available_seasons(*, conn: psycopg.Connection | None = None) -> list[int]:
    """Seasons actually present in the store.

    2019-20 and 2020-21 are deliberately absent (Phase 0 could not pin their
    timezone), so this is the honest source for "what can we train on" rather
    than any hardcoded range.
    """
    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        query = sql.SQL("SELECT DISTINCT season_year FROM {}.lh_game").format(
            sql.Identifier(SCHEMA)
        )
        with conn.cursor() as cur:
            cur.execute(query)
            return sorted(int(row[0]) for row in cur.fetchall())
    finally:
        if owned:
            conn.close()
