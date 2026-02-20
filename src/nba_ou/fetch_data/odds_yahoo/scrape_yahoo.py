from __future__ import annotations

import asyncio
import random
import re
from datetime import date
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

BASE = "https://sports.yahoo.com"
MATCHUP_HREF_RE = re.compile(r"^/nba/.+-\d{10}/?$")
BOARD_ID_FULL = "odds-board-six-pack-FULL"
TIMEOUT_MS = 10_000
TIMEOUT_MS_PUBLIC = 5_200
SLEEP_MIN_S = 1.0
SLEEP_MAX_S = 2.0


def season_year_for_date(d: date) -> int:
    # Yahoo season start year. 2020 was delayed, so threshold is November there.
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
    path = urlparse(odds_url).path
    m = re.search(r"-(\d{8})\d{2}/?$", path)
    if not m:
        raise ValueError(f"Could not parse date from odds_url path: {path}")
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
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
) -> dict[str, object] | None:
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


async def scrape_yahoo_day(
    page: Page,
    d: date,
    *,
    skip_navigation: bool = False,
    target_date: date | None = None,
) -> tuple[pd.DataFrame | None, str]:
    season_year = season_year_for_date(d)
    schedule_url = build_schedule_url(season_year, d)

    if not skip_navigation:
        await page.goto(schedule_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        await try_click_consent(page)

    # Get the actual URL after any redirects
    actual_url = page.url

    try:
        await page.wait_for_selector("tr[rowindex] a[href]", timeout=TIMEOUT_MS)
    except PlaywrightTimeoutError:
        await page.wait_for_selector("table tr a[href]", timeout=TIMEOUT_MS)

    hrefs: list[str] = await page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => e.getAttribute('href')).filter(Boolean)",
    )
    matchup_hrefs = dedupe_keep_order([h for h in hrefs if MATCHUP_HREF_RE.match(h)])
    if not matchup_hrefs:
        return None, actual_url

    matchup_urls = [urljoin(BASE, h) for h in matchup_hrefs]
    daily_dfs: list[pd.DataFrame] = []
    day_str = d.isoformat()

    for matchup_url in matchup_urls:
        odds_url = with_section_odds(matchup_url)

        # Validate URL date matches target date if specified
        if target_date is not None:
            try:
                url_date_str = date_from_matchup_url(odds_url)
                url_date = date.fromisoformat(url_date_str)
                if url_date != target_date:
                    print(
                        f"Skipping {odds_url}: URL date {url_date_str} doesn't match target {target_date.isoformat()}"
                    )
                    continue
            except (ValueError, AttributeError) as e:
                print(f"Failed to parse date from {odds_url}: {e}")
                pass

        await random_sleep()
        await page.goto(odds_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)

        # Check for "No Bets Available" empty state
        try:
            no_bets_section = await page.query_selector("#odds-empty-state")
            if no_bets_section:
                print(f"No bets available for matchup {odds_url}, skipping...")
                continue
        except Exception:
            pass  # If check fails, continue normally

        has_board = True
        try:
            await page.wait_for_selector(
                f"#{BOARD_ID_FULL} table tbody tr", timeout=TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            has_board = False

        if not has_board:
            continue

        df_game = await extract_full_game_odds_df(page, odds_url)
        if df_game.empty:
            continue

        try:
            await page.wait_for_selector(
                "span:has-text('Full Game Public Bet %')", timeout=TIMEOUT_MS_PUBLIC
            )
        except PlaywrightTimeoutError:
            pass

        public = await extract_full_game_public_bet_percentages(page)
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

    if not daily_dfs:
        return None, actual_url

    return pd.concat(daily_dfs, ignore_index=True), actual_url


async def scrape_yahoo_days(
    days: list[date], *, headless: bool = True, target_dates: list[date] | None = None
) -> pd.DataFrame:
    if not days:
        return pd.DataFrame()

    if target_dates is not None and len(target_dates) != len(days):
        raise ValueError(
            f"target_dates length ({len(target_dates)}) must match days length ({len(days)})"
        )

    all_rows: list[pd.DataFrame] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        last_actual_url = None

        for i, d in enumerate(days):
            print(f"Scraping Yahoo for {d.isoformat()}...")
            try:
                # Build the schedule URL for this day
                season_year = season_year_for_date(d)
                current_schedule_url = build_schedule_url(season_year, d)

                # Skip navigation if the actual URL from last scrape matches where we're going
                skip_nav = current_schedule_url == last_actual_url

                # Get target date for validation if provided
                target_d = target_dates[i] if target_dates is not None else None

                df_day, actual_url = await scrape_yahoo_day(
                    page, d, skip_navigation=skip_nav, target_date=target_d
                )
                if df_day is not None and not df_day.empty:
                    all_rows.append(df_day)

                # Update with the actual URL after navigation/redirect
                last_actual_url = actual_url

            except Exception as e:
                print(f"Yahoo scrape failed for {d.isoformat()}: {e}")

            await random_sleep()

        await browser.close()

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


if __name__ == "__main__":
    today = date.today()
    input_dates = [today]

    result_df = asyncio.run(scrape_yahoo_days(input_dates, headless=True))
    if not result_df.empty:
        print(result_df)
