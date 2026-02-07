from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE: str = "https://www.sportsbookreview.com"
HEADLESS: bool = False

# SBR is heavier than Yahoo, so give it time
TIMEOUT_MS: int = 10_000


OUT_DIR: Path = Path(
    "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/sbr_totals_full_game"
)
SAVE_HTML: bool = True
SAVE_CSV: bool = True

SLEEP_MIN_S: float = 0.7
SLEEP_MAX_S: float = 1.6
# =========================


@dataclass(frozen=True)
class RunPaths:
    season_year: int
    season_root: Path
    csv_dir: Path
    html_dir: Path
    csv_moneyline_dir: Path
    html_moneyline_dir: Path
    csv_spread_dir: Path
    html_spread_dir: Path


@dataclass(frozen=True)
class TotalsScrapeResult:
    df: Optional[pd.DataFrame]
    no_odds: bool


def season_year_for_date(d: date) -> int:
    # If you want NBA season-start-year logic, keep this.
    # For a simple folder by calendar year, just return d.year.
    month_threshold = 10
    return d.year if d.month >= month_threshold else (d.year - 1)


def build_sbr_totals_url(d: date) -> str:
    # totals/full-game/ with date=YYYY-MM-DD
    return f"{BASE}/betting-odds/nba-basketball/totals/full-game/?date={d.isoformat()}"


def build_sbr_moneyline_url(d: date) -> str:
    return (
        f"{BASE}/betting-odds/nba-basketball/money-line/full-game/?date={d.isoformat()}"
    )


def build_sbr_spread_url(d: date) -> str:
    return f"{BASE}/betting-odds/nba-basketball/?date={d.isoformat()}"


def ensure_paths_for_season(season_year: int) -> RunPaths:
    season_root = OUT_DIR / f"{season_year}"
    csv_dir = season_root / "csv"
    html_dir = season_root / "html"
    csv_moneyline_dir = season_root / "csv_moneyline"
    html_moneyline_dir = season_root / "html_moneyline"
    csv_spread_dir = season_root / "csv_spread"
    html_spread_dir = season_root / "html_spread"
    for p in [
        csv_dir,
        html_dir,
        csv_moneyline_dir,
        html_moneyline_dir,
        csv_spread_dir,
        html_spread_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        season_year=season_year,
        season_root=season_root,
        csv_dir=csv_dir,
        html_dir=html_dir,
        csv_moneyline_dir=csv_moneyline_dir,
        html_moneyline_dir=html_moneyline_dir,
        csv_spread_dir=csv_spread_dir,
        html_spread_dir=html_spread_dir,
    )


def daterange(d0: date, d1: date) -> List[date]:
    out: List[date] = []
    cur = d0
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out


async def random_sleep(min_s: float = SLEEP_MIN_S, max_s: float = SLEEP_MAX_S) -> None:
    await asyncio.sleep(random.uniform(min_s, max_s))


async def try_click_consent(page: Page) -> None:
    # SBR sometimes shows consent banners (wording varies).
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
    # common normalizations
    slug = slug.replace("logo", "").strip("_")
    return slug


async def extract_sbr_totals_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    """
    Extracts NBA totals full-game odds from SBR for a single date.

    Output is "as presented":
      - 2 rows per game: top row, bottom row
      - On totals pages: top is Over, bottom is Under
      - Includes consensus percentages and consensus opener, plus per-sportsbook line/price
    """
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      // Books in the header, left-to-right
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

      // Each game row has a left container with data-horizontal-eid
      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        // Find an ancestor that contains odds cell anchors (most reliable stable hook)
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
        // expects something like "O 224.5 -110" or "224.5 -110"
        const t = String(txt || "");
        const side = (t.match(/\b([OU])\b/) || [null, null])[1];
        const line = (t.match(/([0-9]+(?:\.[0-9]+)?)/) || [null, null])[1];
        const price = (t.match(/([+-]\d{2,4})/) || [null, null])[1];
        return { side: side || null, line: line || null, price: price || null };
      };

      const parseCellTwoRows = (cell) => {
        // returns [{line, price}, {line, price}] from top/bottom rows
        const out = [
          { line: null, price: null },
          { line: null, price: null }
        ];
        if (!cell) return out;

        // each row is typically a span[role="button"] with "line price"
        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(b => norm(b.textContent)).filter(Boolean);

        // Usually 2 entries: top, bottom
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

        // Time (left header column)
        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        // Matchup URL (use first matchup link)
        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        // Teams: two fw-bolder spans in the left col-3 block
        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        // Scores (optional)
        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        // Consensus % (WAGERS column data-vertical-sbid="-2") has 2 rows of % values
        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        // Consensus opener (OPENER column data-vertical-sbid="-1") has 2 rows "O 224.5 -110" / "U 224.5 -110"
        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openRows = openRowSpans.map(s => parseLineAndPrice(norm(s.textContent))).slice(0, 2);

        // Sportsbook odds cells in the grid (two rows each)
        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));

        // Align cellDivs with books by index; keep only first N to match header books if present
        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRows(c));

        // Build 2 output rows: top/bottom, matching the table layout
        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const openerParsed = openRows[r] || { side: null, line: null, price: null };
          const total_side = openerParsed.side || (r === 0 ? "O" : "U");

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r, // 0 top, 1 bottom
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
    rows: List[Dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Add scraped date fields
    df.insert(0, "date", page_date.isoformat())

    # Numeric cleaning
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

    # Convert sportsbook columns
    for c in df.columns:
        if c.endswith("__line") or c.endswith("__price"):
            df[c] = to_float(df[c])

    # Slugify sportsbook column names (stable downstream columns)
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and ("__line" in c or "__price" in c):
            # book__FanDuel__line -> fanduel_line
            parts = c.split("__")
            # ["book", "<name>", "line"]
            if len(parts) == 3:
                book_name = parts[1]
                suffix = parts[2]
                rename_map[c] = f"{_slugify_book(book_name)}_{suffix}"
    df = df.rename(columns=rename_map)

    # Keep a clean-ish front ordering
    front = [
        "date",
        "event_id",
        "start_time",
        "matchup_url",
        "row_index",
        "team_name",
        "score",
        "consensus_pct",
        "consensus_opener_side",
        "consensus_opener_line",
        "consensus_opener_price",
    ]
    remaining = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + remaining]

    return df


async def extract_sbr_moneyline_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    """
    Extracts NBA moneyline full-game odds from SBR for a single date.

    Output matches the table layout:
      - 2 rows per game (row_index 0 top, 1 bottom)
      - One line per displayed team row
      - consensus_pct (often '-') -> None
      - consensus_opener_price and per-sportsbook prices
    """
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      // Books in the header, left-to-right
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
        const m = t.match(/([+-]\d{2,4})/);
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
    rows: List[Dict[str, Any]] = (payload or {}).get("rows") or []
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

    rename_map: Dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and c.endswith("__price"):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                rename_map[c] = f"{_slugify_book(book_name)}_price"
    df = df.rename(columns=rename_map)

    front = [
        "date",
        "event_id",
        "start_time",
        "matchup_url",
        "row_index",
        "team_row",
        "team_name",
        "score",
        "consensus_pct",
        "consensus_opener_price",
    ]
    remaining = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + remaining]
    return df


async def extract_sbr_spread_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    """
    Extracts NBA spread (main odds page) from SBR for a single date.

    Output matches the table layout:
      - 2 rows per game (row_index 0 top, 1 bottom)
      - One line per displayed team row
      - consensus_pct
      - consensus_opener_spread_line, consensus_opener_spread_price
      - per-sportsbook spread_line + spread_price
    """
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
    rows: List[Dict[str, Any]] = (payload or {}).get("rows") or []
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

    rename_map: Dict[str, str] = {}
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

    front = [
        "date",
        "event_id",
        "start_time",
        "matchup_url",
        "row_index",
        "team_row",
        "team_name",
        "score",
        "consensus_pct",
        "consensus_opener_spread_line",
        "consensus_opener_spread_price",
    ]
    remaining = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + remaining]
    return df


async def scrape_day(page: Page, d: date, paths: RunPaths) -> TotalsScrapeResult:
    url = build_sbr_totals_url(d)
    day_str = d.isoformat()

    html_path = paths.html_dir / f"{day_str}.html"

    loaded_from_html = False
    if SAVE_HTML and html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        await page.set_content(html, wait_until="domcontentloaded")
        loaded_from_html = True
    else:
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        await try_click_consent(page)

    if await has_no_odds_message(page):
        return TotalsScrapeResult(df=None, no_odds=True)

    # Wait for NBA section and at least one game row
    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return TotalsScrapeResult(df=None, no_odds=False)

    # Save full page HTML (only if live)
    if SAVE_HTML and not loaded_from_html:
        html_path.write_text(await page.content(), encoding="utf-8")

    df = await extract_sbr_totals_full_game_rows(page, d)
    if df.empty and loaded_from_html:
        # Fallback to live fetch if cached HTML was incomplete
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        await try_click_consent(page)
        try:
            await page.wait_for_selector(
                "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            return TotalsScrapeResult(df=None, no_odds=False)
        if SAVE_HTML:
            html_path.write_text(await page.content(), encoding="utf-8")
        df = await extract_sbr_totals_full_game_rows(page, d)

    if df.empty:
        return TotalsScrapeResult(df=None, no_odds=False)

    # Add season folder notion
    df.insert(1, "season", paths.season_year)
    df.insert(2, "search_date", day_str)

    return TotalsScrapeResult(df=df, no_odds=False)


async def scrape_day_moneyline(
    page: Page, d: date, paths: RunPaths
) -> Optional[pd.DataFrame]:
    url = build_sbr_moneyline_url(d)
    day_str = d.isoformat()

    html_path = paths.html_moneyline_dir / f"{day_str}.html"
    loaded_from_html = False

    if SAVE_HTML and html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        await page.set_content(html, wait_until="domcontentloaded")
        loaded_from_html = True
    else:
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

    if SAVE_HTML and not loaded_from_html:
        html_path.write_text(await page.content(), encoding="utf-8")

    df = await extract_sbr_moneyline_full_game_rows(page, d)

    if df.empty and loaded_from_html:
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        await try_click_consent(page)
        try:
            await page.wait_for_selector(
                "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            return None
        if SAVE_HTML:
            html_path.write_text(await page.content(), encoding="utf-8")
        df = await extract_sbr_moneyline_full_game_rows(page, d)

    if df.empty:
        return None

    df.insert(1, "season", paths.season_year)
    df.insert(2, "search_date", day_str)
    return df


async def scrape_day_spread(
    page: Page, d: date, paths: RunPaths
) -> Optional[pd.DataFrame]:
    url = build_sbr_spread_url(d)
    day_str = d.isoformat()

    html_path = paths.html_spread_dir / f"{day_str}.html"
    loaded_from_html = False

    if SAVE_HTML and html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        await page.set_content(html, wait_until="domcontentloaded")
        loaded_from_html = True
    else:
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

    if SAVE_HTML and not loaded_from_html:
        html_path.write_text(await page.content(), encoding="utf-8")

    df = await extract_sbr_spread_full_game_rows(page, d)

    if df.empty and loaded_from_html:
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        await try_click_consent(page)
        try:
            await page.wait_for_selector(
                "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            return None
        if SAVE_HTML:
            html_path.write_text(await page.content(), encoding="utf-8")
        df = await extract_sbr_spread_full_game_rows(page, d)

    if df.empty:
        return None

    df.insert(1, "season", paths.season_year)
    df.insert(2, "search_date", day_str)
    return df


async def run_season(dates: Optional[List[date]] = None) -> None:
    all_days = (
        dates if dates is not None else daterange(SEASON_START_DATE, SEASON_END_DATE)
    )
    remaining_days: List[date] = []

    for d in all_days:
        if not SAVE_CSV:
            remaining_days = all_days
            break

        season_year = season_year_for_date(d)
        paths = ensure_paths_for_season(season_year)
        totals_csv_path = paths.csv_dir / f"{d.isoformat()}.csv"
        moneyline_csv_path = paths.csv_moneyline_dir / f"{d.isoformat()}.csv"
        spread_csv_path = paths.csv_spread_dir / f"{d.isoformat()}.csv"
        if (
            totals_csv_path.exists()
            and moneyline_csv_path.exists()
            and spread_csv_path.exists()
        ):
            print(f"Skipping {d.isoformat()} - all CSVs exist")
            continue
        remaining_days.append(d)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()

        for d in tqdm(remaining_days, desc="Days remaining", unit="day"):
            season_year = season_year_for_date(d)
            paths = ensure_paths_for_season(season_year)

            try:
                totals_csv_path = paths.csv_dir / f"{d.isoformat()}.csv"
                moneyline_csv_path = paths.csv_moneyline_dir / f"{d.isoformat()}.csv"
                spread_csv_path = paths.csv_spread_dir / f"{d.isoformat()}.csv"

                no_odds = False
                if SAVE_CSV and not totals_csv_path.exists():
                    totals_result = await scrape_day(page, d, paths)
                    no_odds = totals_result.no_odds
                    if totals_result.df is not None and not totals_result.df.empty:
                        totals_result.df.to_csv(totals_csv_path, index=False)

                if not no_odds and SAVE_CSV and not moneyline_csv_path.exists():
                    df_ml = await scrape_day_moneyline(page, d, paths)
                    if df_ml is not None and not df_ml.empty:
                        df_ml.to_csv(moneyline_csv_path, index=False)

                if not no_odds and SAVE_CSV and not spread_csv_path.exists():
                    df_spread = await scrape_day_spread(page, d, paths)
                    if df_spread is not None and not df_spread.empty:
                        df_spread.to_csv(spread_csv_path, index=False)

            except Exception as e:
                err_path = paths.season_root / "errors.log"
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now().isoformat()} | {d.isoformat()} | {repr(e)}\n"
                    )

            await random_sleep()

        await browser.close()


if __name__ == "__main__":
    SEASON_START_DATE: date = date(2025, 12, 5)
    SEASON_END_DATE: date = date(2026, 2, 4)
    all_days = daterange(SEASON_START_DATE, SEASON_END_DATE)
    # To run for a single specific date, set `specific_day` to that date.
    # Example: date.fromisoformat("2025-10-10")
    # specific_day = date.fromisoformat("2025-12-10")

    # if specific_day:
    #     asyncio.run(run_season([specific_day]))
    # else:
    print("Starting scrape...")
    print(f"Season dates: {SEASON_START_DATE} to {SEASON_END_DATE}")
    asyncio.run(run_season(all_days))
    print("Scrape complete.")
