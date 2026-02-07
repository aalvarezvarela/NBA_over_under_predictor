from __future__ import annotations

import asyncio
from datetime import date

from scrape_yahoo import daterange, run_season


def build_season_date_ranges() -> list[tuple[date, date]]:
    return [
        # (date(2022, 10, 10), date(2023, 7, 1)),
        # (date(2023, 10, 10), date(2024, 7, 1)),
        # (date(2024, 10, 10), date(2025, 7, 1)),
        (date(2025, 12, 30), date(2026, 2, 4)),
    ]


async def run_selected_seasons() -> None:
    for start, end in build_season_date_ranges():
        dates = daterange(start, end)
        await run_season(dates)


if __name__ == "__main__":
    asyncio.run(run_selected_seasons())
