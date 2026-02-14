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
from typing import Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE: str = "https://sports.yahoo.com"
HEADLESS: bool = False
TIMEOUT_MS: int = 4_000
TIMEOUT_MS_PUBLIC: int = 200
# Define the season window you want to scrape (inclusive).
# Example: 2022 season (2022-2023 NBA season) roughly runs Oct 2022 to Jun 2023.
SEASON_START_DATE: date = date(2021, 10, 19)
SEASON_END_DATE: date = date(2022, 7, 1)

OUT_DIR: Path = Path(
    "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/yahoo_odds"
)
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
    # Exception: for the 2020 season use November as the threshold
    month_threshold = 11 if d.year == 2020 else 10
    return d.year if d.month >= month_threshold else (d.year - 1)


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


async def extract_full_game_public_bet_percentages(
    page: Page,
) -> Optional[Dict[str, object]]:
    """
    Extract Full Game Public Bet % table for Spread + Total + Money Line (12 values):
      - spread_pct_bets_away, spread_pct_bets_home
      - spread_pct_money_away, spread_pct_money_home
      - total_pct_bets_over, total_pct_bets_under
      - total_pct_money_over, total_pct_money_under
      - moneyline_pct_bets_away, moneyline_pct_bets_home
      - moneyline_pct_money_away, moneyline_pct_money_home
    """
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const title = Array.from(document.querySelectorAll("span"))
        .find(el => norm(el.textContent) === "Full Game Public Bet %");
      if (!title) return null;

      const section = title.closest("section") || title.closest("div");
      if (!section) return null;

      const tables = Array.from(section.querySelectorAll("table"));
      if (!tables.length) return null;

      const isPublicTable = (table) => {
        const firstCell = table.querySelector("tbody tr td");
        const txt = firstCell ? norm(firstCell.textContent) : "";
        return txt.includes("% OF BETS") || txt.includes("% OF MONEY");
      };

      const table = tables.find(isPublicTable);
      if (!table) return null;

      const ths = Array.from(table.querySelectorAll("thead th"));

      const extractSideLabels = (th) => {
        const spans = Array.from(th.querySelectorAll("span"))
          .map(s => norm(s.textContent))
          .filter(Boolean);

        const sideCandidates = spans.filter(t =>
          /^([OU])\s*\d/.test(t) || /[+-]\d/.test(t)
        );

        if (sideCandidates.length >= 2) {
          const left = sideCandidates[sideCandidates.length - 2];
          const right = sideCandidates[sideCandidates.length - 1];
          return { left, right };
        }
        return { left: null, right: null };
      };

      const spreadTh = ths[1] || null;
      const totalTh  = ths[2] || null;

      const spreadLabels = spreadTh ? extractSideLabels(spreadTh) : { left: null, right: null };
      const totalLabels  = totalTh  ? extractSideLabels(totalTh)  : { left: null, right: null };

      const rows = Array.from(table.querySelectorAll("tbody tr"));

      const getRowKind = (tr) => {
        const td0 = tr.querySelector("td");
        const t = td0 ? norm(td0.textContent) : "";
        if (t.includes("% OF BETS")) return "bets";
        if (t.includes("% OF MONEY")) return "money";
        return null;
      };

      const getTwoPercentsFromCell = (td) => {
        if (!td) return [null, null];
        const txt = norm(td.textContent);
        const matches = Array.from(txt.matchAll(/(\d+(?:\.\d+)?)\s*%/g)).map(m => m[1]);
        if (matches.length >= 2) {
          return [matches[0], matches[1]];
        }
        const pctEls = Array.from(td.querySelectorAll("*"))
          .map(el => norm(el.textContent))
          .filter(t => /%$/.test(t));
        const m2 = pctEls.map(t => (t.match(/(\d+(?:\.\d+)?)\s*%/) || [null, null])[1]).filter(Boolean);
        if (m2.length >= 2) return [m2[0], m2[1]];
        return [null, null];
      };

      const out = {
        spread_pct_bets_away: null,
        spread_pct_bets_home: null,
        spread_pct_money_away: null,
        spread_pct_money_home: null,

        total_pct_bets_over: null,
        total_pct_bets_under: null,
        total_pct_money_over: null,
        total_pct_money_under: null,

        moneyline_pct_bets_away: null,
        moneyline_pct_bets_home: null,
        moneyline_pct_money_away: null,
        moneyline_pct_money_home: null,
      };

      for (const tr of rows) {
        const kind = getRowKind(tr);
        if (!kind) continue;

        const tds = Array.from(tr.querySelectorAll("td"));
        const spreadCell = tds[1] || null;
        const totalCell  = tds[2] || null;
        const moneyCell  = tds[3] || null;

        const [sL, sR] = getTwoPercentsFromCell(spreadCell);
        const [tL, tR] = getTwoPercentsFromCell(totalCell);
        const [mL, mR] = getTwoPercentsFromCell(moneyCell);

        if (kind === "bets") {
          out.spread_pct_bets_away = sL;
          out.spread_pct_bets_home = sR;
          out.total_pct_bets_over = tL;
          out.total_pct_bets_under = tR;
          out.moneyline_pct_bets_away = mL;
          out.moneyline_pct_bets_home = mR;
        } else if (kind === "money") {
          out.spread_pct_money_away = sL;
          out.spread_pct_money_home = sR;
          out.total_pct_money_over = tL;
          out.total_pct_money_under = tR;
          out.moneyline_pct_money_away = mL;
          out.moneyline_pct_money_home = mR;
        }
      }

      return out;
    }
    """
    data = await page.evaluate(js)
    if not data:
        return None

    pct_keys = [
        "spread_pct_bets_away",
        "spread_pct_bets_home",
        "spread_pct_money_away",
        "spread_pct_money_home",
        "total_pct_bets_over",
        "total_pct_bets_under",
        "total_pct_money_over",
        "total_pct_money_under",
        "moneyline_pct_bets_away",
        "moneyline_pct_bets_home",
        "moneyline_pct_money_away",
        "moneyline_pct_money_home",
    ]
    for k in pct_keys:
        v = data.get(k)
        data[k] = float(v) if v is not None else None

    return data


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
        slug = href.strip("/").replace("/", "_")
        html_day_dir = paths.html_dir / day_str
        html_day_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_day_dir / f"{slug}__odds.html"

        loaded_from_html = False
        if SAVE_HTML and html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            await page.set_content(html, wait_until="domcontentloaded")
            loaded_from_html = True
        else:
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
            if loaded_from_html:
                await random_sleep()
                await page.goto(odds_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
                try:
                    await page.wait_for_selector(
                        f"#{BOARD_ID_FULL} table tbody tr", timeout=TIMEOUT_MS
                    )
                    has_board = True
                    loaded_from_html = False
                except PlaywrightTimeoutError:
                    has_board = False

        # Save FULL odds page HTML (only when loaded from live page)
        if SAVE_HTML and not loaded_from_html:
            html_path.write_text(await page.content(), encoding="utf-8")

        # Extract df (only if odds table exists)
        if has_board:
            df_game = await extract_full_game_odds_df(page, odds_url)
            if not df_game.empty:
                try:
                    await page.wait_for_selector(
                        "span:has-text('Full Game Public Bet %')", timeout=TIMEOUT_MS_PUBLIC
                    )
                except PlaywrightTimeoutError:
                    pass

                public = await extract_full_game_public_bet_percentages(page)
                if public is None and loaded_from_html:
                    await random_sleep()
                    await page.goto(
                        odds_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS
                    )
                    loaded_from_html = False
                    try:
                        await page.wait_for_selector(
                            "span:has-text('Full Game Public Bet %')",
                            timeout=TIMEOUT_MS,
                        )
                    except PlaywrightTimeoutError:
                        pass
                    public = await extract_full_game_public_bet_percentages(page)
                    if SAVE_HTML:
                        html_path.write_text(await page.content(), encoding="utf-8")
                if public:
                    for k, v in public.items():
                        df_game[k] = v

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
                # append it if no nulls outside allowed public bet % columns
                allowed_null_cols = {
                    "spread_pct_bets_away",
                    "spread_pct_bets_home",
                    "spread_pct_money_away",
                    "spread_pct_money_home",
                    "total_pct_bets_over",
                    "total_pct_bets_under",
                    "total_pct_money_over",
                    "total_pct_money_under",
                    "moneyline_pct_bets_away",
                    "moneyline_pct_bets_home",
                    "moneyline_pct_money_away",
                    "moneyline_pct_money_home",
                }
                cols_to_check = [c for c in df_game.columns if c not in allowed_null_cols]
                if df_game[cols_to_check].isnull().sum().sum() == 0:
                    daily_dfs.append(df_game)

        if not loaded_from_html:
            await random_sleep()

    if not daily_dfs:
        return None

    return pd.concat(daily_dfs, ignore_index=True)


async def run_season(dates: Optional[List[date]] = None) -> None:
    all_days = dates if dates is not None else daterange(SEASON_START_DATE, SEASON_END_DATE)
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


if __name__ == "__main__":
    all_days = daterange(SEASON_START_DATE, SEASON_END_DATE)
    asyncio.run(run_season(all_days))
