"""One-off repair for spread rows loaded before the price-bleed guard existed.

On a pick'em the SBR cell holds only a price and no spread number, and the
scraper matched that price as the line. Those rows are identifiable structurally
-- a genuine spread satisfies ``left_line = -right_line``, while these carry
complementary price pairs (-110/-110, -115/-105) -- so the fix is exact rather
than heuristic.

The value is relabelled as the price it is; the spread is left NULL rather than
inferred as 0, because the source never stated it.

Lines are stored as doubled half-points, so recovering the price divides by 2.
Prices are whole numbers, so the doubled value is always even and the integer
division is exact.

Idempotent: once repaired, the rows no longer match the predicate.

    python scripts/line_history/repair_spread_price_bleed.py --dry-run
    python scripts/line_history/repair_spread_price_bleed.py
"""

from __future__ import annotations

import argparse

from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven.schema import SCHEMA

SPREAD_MARKET_CODE = "point_spread"

FIND_SQL = f"""
    SELECT count(*),
           count(*) FILTER (WHERE left_price IS NOT NULL OR right_price IS NOT NULL)
    FROM {SCHEMA}.lh_line l
    JOIN {SCHEMA}.lh_market m USING (market_id)
    WHERE m.code = %s
      AND l.left_line IS NOT NULL
      AND l.right_line IS NOT NULL
      AND l.left_line <> -l.right_line
"""

REPAIR_SQL = f"""
    UPDATE {SCHEMA}.lh_line l
    SET left_price  = COALESCE(l.left_price,  l.left_line  / 2),
        right_price = COALESCE(l.right_price, l.right_line / 2),
        left_line   = NULL,
        right_line  = NULL
    FROM {SCHEMA}.lh_market m
    WHERE m.market_id = l.market_id
      AND m.code = %s
      AND l.left_line IS NOT NULL
      AND l.right_line IS NOT NULL
      AND l.left_line <> -l.right_line
"""

VERIFY_SQL = f"""
    SELECT count(*) FILTER (WHERE left_line = -right_line) AS mirrored,
           count(*) FILTER (WHERE left_line <> -right_line) AS still_bad,
           count(*) FILTER (WHERE left_line IS NULL AND left_price IS NOT NULL) AS price_only
    FROM {SCHEMA}.lh_line l
    JOIN {SCHEMA}.lh_market m USING (market_id)
    WHERE m.code = %s
"""


# Pre-game lines outside what the market can produce (a dropped decimal turns
# 228.5 into 2285). In-play rows are exempt -- a live spread really does blow
# out past 30 during a rout. Bounds mirror transform.PREGAME_LINE_BOUNDS.
IMPLAUSIBLE_FIND_SQL = f"""
    SELECT m.code, count(*)
    FROM {SCHEMA}.lh_line l
    JOIN {SCHEMA}.lh_market m USING (market_id)
    WHERE l.is_pregame
      AND (
            (m.code = 'totals' AND l.left_line IS NOT NULL
             AND (l.left_line / 2.0 < 150 OR l.left_line / 2.0 > 300))
         OR (m.code = 'point_spread' AND l.left_line IS NOT NULL
             AND abs(l.left_line / 2.0) > 30)
      )
    GROUP BY 1
"""

IMPLAUSIBLE_REPAIR_SQL = f"""
    UPDATE {SCHEMA}.lh_line l
    SET left_line = NULL, right_line = NULL
    FROM {SCHEMA}.lh_market m
    WHERE m.market_id = l.market_id
      AND l.is_pregame
      AND (
            (m.code = 'totals' AND l.left_line IS NOT NULL
             AND (l.left_line / 2.0 < 150 OR l.left_line / 2.0 > 300))
         OR (m.code = 'point_spread' AND l.left_line IS NOT NULL
             AND abs(l.left_line / 2.0) > 30)
      )
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = connect_line_history_db()
    try:
        with conn.cursor() as cur:
            cur.execute(FIND_SQL, (SPREAD_MARKET_CODE,))
            affected, with_existing_price = cur.fetchone()
            cur.execute(IMPLAUSIBLE_FIND_SQL)
            implausible = cur.fetchall()

        print(f"Spread rows with a price in the line column: {affected:,}")
        if with_existing_price:
            print(
                f"  of which {with_existing_price:,} already carry a price "
                "(kept; only the NULL ones are filled)"
            )
        total_implausible = sum(count for _code, count in implausible)
        print(f"Pre-game rows with an impossible line: {total_implausible:,}")
        for code, count in implausible:
            print(f"  {code}: {count:,}")

        if affected == 0 and total_implausible == 0:
            print("Nothing to repair.")
            return 0

        if args.dry_run:
            print("DRY RUN: no changes written.")
            return 0

        with conn.cursor() as cur:
            cur.execute(REPAIR_SQL, (SPREAD_MARKET_CODE,))
            updated = cur.rowcount
            cur.execute(IMPLAUSIBLE_REPAIR_SQL)
            cleared = cur.rowcount
        conn.commit()
        print(f"Relabelled {updated:,} price-bleed rows.")
        print(f"Cleared {cleared:,} impossible pre-game lines.")

        with conn.cursor() as cur:
            cur.execute(VERIFY_SQL, (SPREAD_MARKET_CODE,))
            mirrored, still_bad, price_only = cur.fetchone()
            cur.execute(IMPLAUSIBLE_FIND_SQL)
            remaining = sum(count for _code, count in cur.fetchall())
        print(
            f"Verify -> mirrored {mirrored:,} | still invalid {still_bad:,} "
            f"| price-only {price_only:,} | implausible {remaining:,}"
        )
        return 0 if still_bad == 0 and remaining == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
