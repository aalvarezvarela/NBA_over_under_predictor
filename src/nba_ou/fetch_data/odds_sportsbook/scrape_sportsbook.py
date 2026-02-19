import asyncio
import random
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from nba_ou.fetch_data.odds_sportsbook.process_money_line_data import (
    load_one_day_moneyline_csv,
)
from nba_ou.fetch_data.odds_sportsbook.process_spread_data import (
    load_one_day_spread_csv,
)
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    load_one_day_totals_csv,
)
from nba_ou.postgre_db.odds_sportsbook.process_sportsbook_data import merge_daily_frames
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

BASE_URL = "https://www.sportsbookreview.com"
TIMEOUT_MS = 10_000
SLEEP_MIN_S = 0.7
SLEEP_MAX_S = 1.6


@dataclass(frozen=True)
class TotalsScrapeResult:
    df: pd.DataFrame | None
    no_odds: bool


def season_year_for_date(d: date) -> int:
    # NBA season start year (e.g. Jan 2026 belongs to 2025-26 => 2025)
    return d.year if d.month >= 10 else (d.year - 1)


def build_sbr_totals_url(d: date) -> str:
    return (
        f"{BASE_URL}/betting-odds/nba-basketball/totals/full-game/?date={d.isoformat()}"
    )


def build_sbr_moneyline_url(d: date) -> str:
    return f"{BASE_URL}/betting-odds/nba-basketball/money-line/full-game/?date={d.isoformat()}"


def build_sbr_spread_url(d: date) -> str:
    return f"{BASE_URL}/betting-odds/nba-basketball/?date={d.isoformat()}"


async def random_sleep(min_s: float = SLEEP_MIN_S, max_s: float = SLEEP_MAX_S) -> None:
    await asyncio.sleep(random.uniform(min_s, max_s))


async def try_click_consent(page: Page) -> None:
    candidates = [
        "Accept all",
        "I agree",
        "Agree",
        "Accept",
        "Aceptar",
        "Aceptar todo",
        "Continue",
        "OK",
        "Got it",
    ]
    for label in candidates:
        try:
            await page.get_by_role("button", name=label).click(timeout=500)
            return
        except Exception:
            pass


async def has_no_odds_message(page: Page) -> bool:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
      const els = Array.from(document.querySelectorAll("div"));
      return els.some(el => norm(el.textContent) === "No odds available at this time for this league");
    }
    """
    try:
        return bool(await page.evaluate(js))
    except Exception:
        return False


def _slugify_book(book: str) -> str:
    s = book.strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    slug = "".join(out).strip("_")
    return slug.replace("logo", "").strip("_")


async def extract_sbr_totals_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      const thead = section.querySelector("#thead-nba");
      let books = [];
      if (thead) {
        const imgs = Array.from(thead.querySelectorAll("a img[alt]"));
        books = imgs
          .map(img => norm(img.getAttribute("alt")))
          .map(t => t.replace(/\s+Logo$/i, ""))
          .filter(Boolean);
      }

      const tbody = section.querySelector("#tbody-nba");
      if (!tbody) return { books, rows: [] };

      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const parsePct = (txt) => {
        const m = String(txt || "").match(/(\d+(?:\.\d+)?)\s*%/);
        return m ? m[1] : null;
      };

      const parseLineAndPrice = (txt) => {
        const t = String(txt || "");
        const side = (t.match(/\b([OU])\b/) || [null, null])[1];
        const line = (t.match(/([0-9]+(?:\.[0-9]+)?)/) || [null, null])[1];
        const price = (t.match(/([+-]\d{2,4})/) || [null, null])[1];
        return { side: side || null, line: line || null, price: price || null };
      };

      const parseCellTwoRows = (cell) => {
        const out = [
          { line: null, price: null },
          { line: null, price: null }
        ];
        if (!cell) return out;

        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(b => norm(b.textContent)).filter(Boolean);

        for (let i = 0; i < Math.min(2, texts.length); i++) {
          const parsed = parseLineAndPrice(texts[i]);
          out[i] = { line: parsed.line, price: parsed.price };
        }
        return out;
      };

      const rows = [];

      for (const eidNode of leftEids) {
        const eidRaw = eidNode.getAttribute("data-horizontal-eid");
        const event_id = eidRaw ? Number(eidRaw) : null;

        const root = ascendToGameRoot(eidNode);

        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openRows = openRowSpans.map(s => parseLineAndPrice(norm(s.textContent))).slice(0, 2);

        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));

        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRows(c));

        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const openerParsed = openRows[r] || { side: null, line: null, price: null };
          const total_side = openerParsed.side || (r === 0 ? "O" : "U");

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r,
            team_name,
            score,
            consensus_pct: cons_pcts[r] || null,
            consensus_opener_side: total_side,
            consensus_opener_line: openerParsed.line || null,
            consensus_opener_price: openerParsed.price || null,
          };

          for (let j = 0; j < n; j++) {
            const book = books[j] || `book_${j}`;
            const book_key = book;
            const two = cells[j] || [{line:null,price:null},{line:null,price:null}];
            const v = two[r] || { line: null, price: null };
            row[`book__${book_key}__line`] = v.line;
            row[`book__${book_key}__price`] = v.price;
          }

          rows.push(row);
        }
      }

      return { books, rows };
    }
    """

    payload = await page.evaluate(js)
    rows: list[dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())

    def to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string").str.replace("−", "-", regex=False), errors="coerce"
        )

    df["consensus_pct"] = to_float(df.get("consensus_pct", pd.Series(dtype="string")))
    df["consensus_opener_line"] = to_float(
        df.get("consensus_opener_line", pd.Series(dtype="string"))
    )
    df["consensus_opener_price"] = to_float(
        df.get("consensus_opener_price", pd.Series(dtype="string"))
    )

    for c in df.columns:
        if c.endswith("__line") or c.endswith("__price"):
            df[c] = to_float(df[c])

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and ("__line" in c or "__price" in c):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                suffix = parts[2]
                rename_map[c] = f"{_slugify_book(book_name)}_{suffix}"
    df = df.rename(columns=rename_map)

    return df


async def extract_sbr_moneyline_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      const thead = section.querySelector("#thead-nba");
      let books = [];
      if (thead) {
        const imgs = Array.from(thead.querySelectorAll("a img[alt]"));
        books = imgs
          .map(img => norm(img.getAttribute("alt")))
          .map(t => t.replace(/\s+Logo$/i, ""))
          .filter(Boolean);
      }

      const tbody = section.querySelector("#tbody-nba");
      if (!tbody) return { books, rows: [] };

      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const parsePct = (txt) => {
        const t = String(txt || "");
        const m = t.match(/(\d+(?:\.\d+)?)\s*%/);
        return m ? m[1] : null;
      };

      const parsePrice = (txt) => {
        const t = String(txt || "");
        const m = t.match(/([+-]\d{2,5})/);
        return m ? m[1] : null;
      };

      const parseCellTwoRowsPrice = (cell) => {
        const out = [ { price: null }, { price: null } ];
        if (!cell) return out;

        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(b => norm(b.textContent)).filter(Boolean);

        for (let i = 0; i < Math.min(2, texts.length); i++) {
          out[i] = { price: parsePrice(texts[i]) };
        }
        return out;
      };

      const rows = [];

      for (const eidNode of leftEids) {
        const eidRaw = eidNode.getAttribute("data-horizontal-eid");
        const event_id = eidRaw ? Number(eidRaw) : null;

        const root = ascendToGameRoot(eidNode);

        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openPrices = openRowSpans.map(s => parsePrice(norm(s.textContent))).slice(0, 2);

        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));
        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRowsPrice(c));

        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r,
            team_row: (r === 0 ? "away" : "home"),
            team_name,
            score,
            consensus_pct: cons_pcts[r] || null,
            consensus_opener_price: openPrices[r] || null,
          };

          for (let j = 0; j < n; j++) {
            const book = books[j] || `book_${j}`;
            const two = cells[j] || [{price:null},{price:null}];
            const v = two[r] || { price: null };
            row[`book__${book}__price`] = v.price;
          }

          rows.push(row);
        }
      }

      return { books, rows };
    }
    """

    payload = await page.evaluate(js)
    rows: list[dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())

    def to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string").str.replace("−", "-", regex=False), errors="coerce"
        )

    df["consensus_pct"] = to_float(df.get("consensus_pct", pd.Series(dtype="string")))
    df["consensus_opener_price"] = to_float(
        df.get("consensus_opener_price", pd.Series(dtype="string"))
    )

    for c in df.columns:
        if c.endswith("__price") and c.startswith("book__"):
            df[c] = to_float(df[c])

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and c.endswith("__price"):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                rename_map[c] = f"{_slugify_book(book_name)}_price"
    df = df.rename(columns=rename_map)
    return df


async def extract_sbr_spread_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      const thead = section.querySelector("#thead-nba");
      let books = [];
      if (thead) {
        const imgs = Array.from(thead.querySelectorAll("a img[alt]"));
        books = imgs
          .map(img => norm(img.getAttribute("alt")))
          .map(t => t.replace(/\s+Logo$/i, ""))
          .filter(Boolean);
      }

      const tbody = section.querySelector("#tbody-nba");
      if (!tbody) return { books, rows: [] };

      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const parsePct = (txt) => {
        const t = String(txt || "");
        const m = t.match(/(\d+(?:\.\d+)?)\s*%/);
        return m ? m[1] : null;
      };

      const parseLineAndPrice = (txt) => {
        const t = String(txt || "");
        const line = (t.match(/([+-]\d+(?:\.\d+)?)/) || [null, null])[1];
        const price = (t.match(/([+-]\d{2,4})/) || [null, null])[1];
        return { line: line || null, price: price || null };
      };

      const parseCellTwoRows = (cell) => {
        const out = [
          { line: null, price: null },
          { line: null, price: null }
        ];
        if (!cell) return out;

        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(b => norm(b.textContent)).filter(Boolean);

        for (let i = 0; i < Math.min(2, texts.length); i++) {
          out[i] = parseLineAndPrice(texts[i]);
        }
        return out;
      };

      const rows = [];

      for (const eidNode of leftEids) {
        const eidRaw = eidNode.getAttribute("data-horizontal-eid");
        const event_id = eidRaw ? Number(eidRaw) : null;

        const root = ascendToGameRoot(eidNode);

        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openRows = openRowSpans.map(s => parseLineAndPrice(norm(s.textContent))).slice(0, 2);

        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));
        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRows(c));

        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const opener = openRows[r] || { line: null, price: null };

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r,
            team_row: (r === 0 ? "away" : "home"),
            team_name,
            score,
            consensus_pct: cons_pcts[r] || null,
            consensus_opener_spread_line: opener.line,
            consensus_opener_spread_price: opener.price,
          };

          for (let j = 0; j < n; j++) {
            const book = books[j] || `book_${j}`;
            const two = cells[j] || [{line:null,price:null},{line:null,price:null}];
            const v = two[r] || { line: null, price: null };
            row[`book__${book}__spread_line`] = v.line;
            row[`book__${book}__spread_price`] = v.price;
          }

          rows.push(row);
        }
      }

      return { books, rows };
    }
    """

    payload = await page.evaluate(js)
    rows: list[dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())

    def to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string").str.replace("−", "-", regex=False), errors="coerce"
        )

    df["consensus_pct"] = to_float(df.get("consensus_pct", pd.Series(dtype="string")))
    df["consensus_opener_spread_line"] = to_float(
        df.get("consensus_opener_spread_line", pd.Series(dtype="string"))
    )
    df["consensus_opener_spread_price"] = to_float(
        df.get("consensus_opener_spread_price", pd.Series(dtype="string"))
    )

    for c in df.columns:
        if c.startswith("book__") and (
            c.endswith("__spread_line") or c.endswith("__spread_price")
        ):
            df[c] = to_float(df[c])

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and (
            c.endswith("__spread_line") or c.endswith("__spread_price")
        ):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                suffix = parts[2]
                rename_map[c] = f"{_slugify_book(book_name)}_{suffix}"
    df = df.rename(columns=rename_map)
    return df


async def scrape_day_totals(page: Page, d: date) -> TotalsScrapeResult:
    url = build_sbr_totals_url(d)

    await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    if await has_no_odds_message(page):
        return TotalsScrapeResult(df=None, no_odds=True)

    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return TotalsScrapeResult(df=None, no_odds=False)

    df = await extract_sbr_totals_full_game_rows(page, d)
    if df.empty:
        return TotalsScrapeResult(df=None, no_odds=False)

    df.insert(1, "season", season_year_for_date(d))
    return TotalsScrapeResult(df=df, no_odds=False)


async def scrape_day_moneyline(page: Page, d: date) -> pd.DataFrame | None:
    url = build_sbr_moneyline_url(d)

    await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    if await has_no_odds_message(page):
        return None

    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return None

    df = await extract_sbr_moneyline_full_game_rows(page, d)
    if df.empty:
        return None

    df.insert(1, "season", season_year_for_date(d))
    return df


async def scrape_day_spread(page: Page, d: date) -> pd.DataFrame | None:
    url = build_sbr_spread_url(d)

    await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    if await has_no_odds_message(page):
        return None

    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return None

    df = await extract_sbr_spread_full_game_rows(page, d)
    if df.empty:
        return None

    df.insert(1, "season", season_year_for_date(d))
    return df


async def scrape_sportsbook_days(
    days: list[date],
    *,
    headless: bool = True,
) -> pd.DataFrame:
    if not days:
        return pd.DataFrame()

    merged_days: list[pd.DataFrame] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        for d in days:
            try:
                totals_result = await scrape_day_totals(page, d)
                if totals_result.no_odds or totals_result.df is None:
                    await random_sleep()
                    continue

                spread_df_raw = await scrape_day_spread(page, d)
                ml_df_raw = await scrape_day_moneyline(page, d)

                totals_game_df = load_one_day_totals_csv(totals_result.df)
                spread_game_df = (
                    load_one_day_spread_csv(spread_df_raw)
                    if spread_df_raw is not None
                    else pd.DataFrame(columns=["game_id"])
                )
                ml_game_df = (
                    load_one_day_moneyline_csv(ml_df_raw)
                    if ml_df_raw is not None
                    else pd.DataFrame(columns=["game_id"])
                )

                merged_day = merge_daily_frames(
                    totals_game_df, spread_game_df, ml_game_df
                )
                merged_days.append(merged_day)

            except Exception as e:
                print(f"Failed sportsbook scrape for {d.isoformat()}: {e}")

            await random_sleep()

        await browser.close()

    if not merged_days:
        return pd.DataFrame()

    return pd.concat(merged_days, ignore_index=True)
