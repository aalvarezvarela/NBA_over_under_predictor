"""
Yahoo NBA odds scraper (Full Game odds extraction + save FULL odds page HTML)

What it does
- Loops through an entire season day-by-day (SEASON_START_DATE..SEASON_END_DATE, inclusive).
- For each date:
  1) Opens Yahoo schedule page for that date
  2) Extracts matchup links
  3) Opens each matchup odds page (?section=odds)
  4) Extracts Full Game odds (team, spread, total, money line)
  5) Saves the FULL odds page HTML (page.content()) per matchup
  6) Saves a CSV for that date under: OUT_DIR/season=YYYY/csv/YYYY-MM-DD.csv
- Random sleep 1..2 seconds between navigations.

Outputs
- CSV per day: OUT_DIR/season=YYYY/csv/YYYY-MM-DD.csv
- HTML per matchup: OUT_DIR/season=YYYY/html/YYYY-MM-DD/<matchup_slug>__odds.html
- Errors log: OUT_DIR/season=YYYY/errors.log
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

print("Waiting 6 hours before starting scraper to avoid rate limits...")
time.sleep(60 * 60 * 6)
# =========================
# CONFIG
# =========================
BASE: str = "https://sports.yahoo.com"
HEADLESS: bool = False
TIMEOUT_MS: int = 45_000

# Define the season window you want to scrape (inclusive).
# Example: 2022 season (2022-2023 NBA season) roughly runs Oct 2022 to Jun 2023.
SEASON_START_DATE: date = date(2020, 10, 19)
SEASON_END_DATE: date = date(2021, 9, 30)

OUT_DIR: Path = Path("/media/adrian_alvarez/TOSHIBA EXT/NBA_yahoo_odds/")
SAVE_HTML: bool = True
SAVE_CSV: bool = True

SLEEP_MIN_S: float = 1.0
SLEEP_MAX_S: float = 2.0
# =========================

MATCHUP_HREF_RE = re.compile(r"^/nba/.+-\d{10}/?$")
BOARD_ID_FULL = "odds-board-six-pack-FULL"


@dataclass(frozen=True)
class RunPaths:
    season_year: int
    season_root: Path
    csv_dir: Path
    html_dir: Path


def season_year_for_date(d: date) -> int:
    # Yahoo "season" param uses the season start year. NBA seasons start around Oct.
    return d.year if d.month >= 10 else (d.year - 1)


def build_schedule_url(season_year: int, d: date) -> str:
    return f"{BASE}/nba/schedule/?season={season_year}&date={d.isoformat()}"


def with_section_odds(url: str) -> str:
    parts = urlparse(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q["section"] = "odds"
    new_query = urlencode(q)
    return urlunparse(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            parts.params,
            new_query,
            parts.fragment,
        )
    )


def date_from_matchup_url(odds_url: str) -> str:
    # ...-YYYYMMDDxx/ -> YYYY-MM-DD
    path = urlparse(odds_url).path
    m = re.search(r"-(\d{8})\d{2}/?$", path)
    if not m:
        raise ValueError(f"Could not parse date from odds_url path: {path}")
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


async def random_sleep(min_s: float = SLEEP_MIN_S, max_s: float = SLEEP_MAX_S) -> None:
    await asyncio.sleep(random.uniform(min_s, max_s))


async def try_click_consent(page: Page) -> None:
    candidates = ["Accept all", "I agree", "Agree", "Accept", "Aceptar", "Aceptar todo"]
    for label in candidates:
        try:
            await page.get_by_role("button", name=label).click(timeout=1500)
            return
        except Exception:
            pass


async def extract_full_game_odds_df(page: Page, odds_url: str) -> pd.DataFrame:
    selector = f"#{BOARD_ID_FULL} table tbody tr"

    rows = await page.eval_on_selector_all(
        selector,
        r"""
        rows => rows.map(row => {
            const tds = Array.from(row.querySelectorAll("td"));
            const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
            const getAllText = (el) => norm(el ? el.textContent : "");

            const extractFirst = (text, re) => {
                const m = text.match(re);
                return m ? m[1] : null;
            };

            // TEAM: prefer dedicated spans; fallback to img alt; fallback to split trailing abbr
            const teamTd = tds[0];

            const nameEl = teamTd ? teamTd.querySelector("span._ys_1dnwiv5") : null;
            const abbrEl = teamTd ? teamTd.querySelector("span._ys_52mm7a") : null;

            let team_name = nameEl ? norm(nameEl.textContent) : null;
            let team_abbr = abbrEl ? norm(abbrEl.textContent) : null;

            if (!team_name && teamTd) {
                const img = teamTd.querySelector("img[alt]");
                if (img) team_name = norm(img.getAttribute("alt"));
            }

            if ((!team_name || !team_abbr) && teamTd) {
                const raw = getAllText(teamTd);
                const m2 = raw.match(/^(.*?)([A-Z]{2,3})$/);
                if (m2) {
                    if (!team_name) team_name = norm(m2[1]);
                    if (!team_abbr) team_abbr = m2[2];
                } else if (!team_name) {
                    team_name = raw;
                }
            }

            // ODDS CELLS (cells contain explanatory text; extract first numeric token)
            const spread_text = getAllText(tds[1]);
            const total_text  = getAllText(tds[2]);
            const ml_text     = getAllText(tds[3]);

            const spread = extractFirst(spread_text, /([+-]\d+(?:\.\d+)?)/);
            const total_side = extractFirst(total_text, /\b([OU])\b/);
            const total_line = extractFirst(total_text, /[OU]\s*([0-9]+(?:\.[0-9]+)?)/);
            const money_line = extractFirst(ml_text, /([+-]\d+)/);

            return { team_name, team_abbr, spread, total_side, total_line, money_line };
        })
        """,
    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.insert(0, "date", date_from_matchup_url(odds_url))

    for c in ["spread", "total_line", "money_line"]:
        df[c] = df[c].astype("string").str.replace("−", "-", regex=False).str.strip()

    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df["total_line"] = pd.to_numeric(df["total_line"], errors="coerce")
    df["money_line"] = pd.to_numeric(df["money_line"], errors="coerce")

    df["team_name"] = df["team_name"].astype("string").str.strip()
    df["team_abbr"] = df["team_abbr"].astype("string").str.strip()

    return df[
        [
            "date",
            "team_name",
            "team_abbr",
            "spread",
            "total_side",
            "total_line",
            "money_line",
        ]
    ]


def ensure_paths_for_season(season_year: int) -> RunPaths:
    season_root = OUT_DIR / f"{season_year}"
    csv_dir = season_root / "csv"
    html_dir = season_root / "html"
    csv_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        season_year=season_year,
        season_root=season_root,
        csv_dir=csv_dir,
        html_dir=html_dir,
    )


def daterange(d0: date, d1: date) -> List[date]:
    out: List[date] = []
    cur = d0
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out


async def scrape_day(page: Page, d: date, paths: RunPaths) -> Optional[pd.DataFrame]:
    season_year = paths.season_year
    schedule_url = build_schedule_url(season_year, d)

    await page.goto(schedule_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    try:
        await page.wait_for_selector("tr[rowindex] a[href]", timeout=TIMEOUT_MS)
    except PlaywrightTimeoutError:
        await page.wait_for_selector("table tr a[href]", timeout=TIMEOUT_MS)

    hrefs: List[str] = await page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => e.getAttribute('href')).filter(Boolean)",
    )
    matchup_hrefs = dedupe_keep_order([h for h in hrefs if MATCHUP_HREF_RE.match(h)])
    if not matchup_hrefs:
        return None

    matchup_urls = [urljoin(BASE, h) for h in matchup_hrefs]

    daily_dfs: List[pd.DataFrame] = []
    day_str = d.isoformat()

    for href, matchup_url in zip(matchup_hrefs, matchup_urls):
        odds_url = with_section_odds(matchup_url)

        await random_sleep()
        await page.goto(odds_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)

        # Ensure the Full Game odds board exists; if not, still save HTML but skip df
        has_board = True
        try:
            await page.wait_for_selector(
                f"#{BOARD_ID_FULL} table tbody tr", timeout=TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            has_board = False

        # Save FULL odds page HTML
        if SAVE_HTML:
            slug = href.strip("/").replace("/", "_")
            html_day_dir = paths.html_dir / day_str
            html_day_dir.mkdir(parents=True, exist_ok=True)
            html_path = html_day_dir / f"{slug}__odds.html"
            html_path.write_text(await page.content(), encoding="utf-8")

        # Extract df (only if odds table exists)
        if has_board:
            df_game = await extract_full_game_odds_df(page, odds_url)
            if not df_game.empty:
                teams = (
                    df_game.get("team_name", pd.Series(dtype="string"))
                    .dropna()
                    .astype("string")
                    .tolist()
                )
                matchup = f"{teams[0]} @ {teams[1]}" if len(teams) >= 2 else None
                df_game.insert(1, "season", season_year)
                df_game.insert(2, "search_date", day_str)
                df_game.insert(3, "matchup", matchup)
                daily_dfs.append(df_game)

        await random_sleep()

    if not daily_dfs:
        return None

    return pd.concat(daily_dfs, ignore_index=True)


async def run_season() -> None:
    all_days = daterange(SEASON_START_DATE, SEASON_END_DATE)
    remaining_days: List[date] = []

    for d in all_days:
        if not SAVE_CSV:
            remaining_days = all_days
            break

        season_year = season_year_for_date(d)
        paths = ensure_paths_for_season(season_year)
        csv_path = paths.csv_dir / f"{d.isoformat()}.csv"
        if csv_path.exists():
            continue
        remaining_days.append(d)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()

        for d in tqdm(remaining_days, desc="Days remaining", unit="day"):
            season_year = season_year_for_date(d)
            paths = ensure_paths_for_season(season_year)

            csv_path = paths.csv_dir / f"{d.isoformat()}.csv"
            if SAVE_CSV and csv_path.exists():
                continue

            try:
                df_day = await scrape_day(page, d, paths)
                if df_day is None or df_day.empty:
                    continue

                if SAVE_CSV:
                    df_day.to_csv(csv_path, index=False)

            except Exception as e:
                err_path = paths.season_root / "errors.log"
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now().isoformat()} | {d.isoformat()} | {repr(e)}\n"
                    )

            await random_sleep()

        await browser.close()


asyncio.run(run_season())
