"""SBR NBA line history season downloader."""

import asyncio
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import (
    BASE_URL,
    SLEEP_MAX_S,
    SLEEP_MIN_S,
    TIMEOUT_MS,
    build_sbr_moneyline_url,
    build_sbr_spread_url,
    build_sbr_totals_url,
    random_sleep,
    season_year_for_date,
)
from playwright.async_api import Page, Route, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

MARKET_FORMATS: dict[str, str] = {
    "point_spread": "pointspread",
    "money_line": "money-line",
    "totals": "totals",
}
CONSENT_BUTTON_LABELS = [
    "Allow all cookies",
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
BLOCKED_RESOURCE_TYPES = {"font", "media"}

SOURCE_MARKET_URL_BUILDERS = {
    "point_spread": build_sbr_spread_url,
    "money_line": build_sbr_moneyline_url,
    "totals": build_sbr_totals_url,
}

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "sbr_line_history"
LINE_HISTORY_COLUMNS = [
    "game_date",
    "season_year",
    "event_id",
    "start_time",
    "matchup_url",
    "line_history_url",
    "team_away",
    "team_home",
    "bookmaker",
    "bookmaker_slug",
    "market",
    "row_kind",
    "change_order",
    "timestamp_raw",
    "timestamp",
    "left_label",
    "right_label",
    "left_value_raw",
    "right_value_raw",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]
MATCHUP_RECORD_COLUMNS = [
    "game_date",
    "season_year",
    "event_id",
    "start_time",
    "matchup_url",
    "team_away",
    "team_home",
]


@dataclass(frozen=True)
class GameMatchup:
    event_id: str
    game_date: date
    season_year: int
    start_time: str | None
    matchup_url: str
    line_history_url: str
    team_away: str | None
    team_home: str | None


@dataclass(frozen=True)
class DailyOutputPaths:
    season_root: Path
    line_history_dir: Path
    matchup_records_dir: Path
    line_history_csv: Path
    matchup_records_csv: Path
    errors_log: Path


def build_sbr_line_history_url(event_id: str | int) -> str:
    return f"{BASE_URL}/betting-odds/nba-basketball/line-history/{event_id}/"


async def try_click_line_history_consent(page: Page) -> None:
    try:
        clicked = await page.evaluate(
            r"""
            (labels) => {
              const norm = (s) => (s || "").replace(/\s+/g, " ").trim().toLowerCase();
              const wanted = new Set(labels.map(norm));
              for (const button of Array.from(document.querySelectorAll("button"))) {
                if (wanted.has(norm(button.innerText || button.textContent))) {
                  button.click();
                  return true;
                }
              }
              return false;
            }
            """,
            CONSENT_BUTTON_LABELS,
        )
        if clicked:
            await page.wait_for_timeout(100)
    except Exception:
        pass


async def block_heavy_assets(route: Route) -> None:
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


def _slugify_bookmaker(bookmaker: str) -> str:
    s = bookmaker.replace("Logo", "").strip().lower()
    out: list[str] = []
    previous_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            previous_us = False
        elif not previous_us:
            out.append("_")
            previous_us = True

    return "".join(out).strip("_")


def _slugify_matchup_key(value: str) -> str:
    s = (
        value.replace("%", " pct ")
        .replace("&", " and ")
        .replace("+", " plus ")
        .strip()
        .lower()
    )
    out: list[str] = []
    previous_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            previous_us = False
        elif not previous_us:
            out.append("_")
            previous_us = True

    return "".join(out).strip("_")


def _absolute_sbr_url(url: str) -> str:
    if url.startswith("http"):
        return url
    return f"{BASE_URL}{url}"


def _parse_history_datetime(raw_time: str | None, game_date: date) -> datetime | None:
    if not raw_time:
        return None

    normalized = raw_time.strip()
    try:
        parsed = datetime.strptime(
            f"{game_date.year}/{normalized}", "%Y/%m/%d %I:%M %p"
        )
    except ValueError:
        return None

    if parsed.date() > game_date + timedelta(days=30):
        parsed = parsed.replace(year=parsed.year - 1)
    elif parsed.date() < game_date - timedelta(days=330):
        parsed = parsed.replace(year=parsed.year + 1)

    return parsed


def _parse_numeric_token(token: str | None) -> float | None:
    if token is None:
        return None

    normalized = token.strip().replace("−", "-").upper()
    if not normalized or normalized == "-":
        return None
    if normalized in {"PK", "PICK", "PICKEM", "PICK'EM"}:
        return 0.0
    if normalized in {"EV", "EVEN"}:
        return 100.0

    try:
        return float(normalized)
    except ValueError:
        return None


def _parse_odds_value(
    value_raw: str | None, market: str
) -> tuple[float | None, float | None]:
    if value_raw is None:
        return None, None

    normalized = " ".join(value_raw.replace("−", "-").split())
    if not normalized or normalized == "-":
        return None, None

    if market == "money_line":
        return None, _parse_numeric_token(normalized)

    pieces = normalized.split()
    if len(pieces) == 1:
        return _parse_numeric_token(pieces[0]), None

    line_text = " ".join(pieces[:-1])
    price_text = pieces[-1]
    return _parse_numeric_token(line_text), _parse_numeric_token(price_text)


async def extract_matchups_from_odds_page(
    page: Page,
    page_date: date,
    *,
    source_market: str = "totals",
) -> list[GameMatchup]:
    url_builder = SOURCE_MARKET_URL_BUILDERS[source_market]
    await page.goto(
        url_builder(page_date), wait_until="domcontentloaded", timeout=TIMEOUT_MS
    )
    await try_click_line_history_consent(page)

    try:
        await page.wait_for_selector(
            '#section-nba #tbody-nba a[href*="/scores/nba-basketball/matchup/"]',
            timeout=TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        return []

    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
      const section = document.querySelector("#section-nba");
      const tbody = section && section.querySelector("#tbody-nba");
      if (!section || !tbody) return [];

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasMatchup = cur.querySelector && cur.querySelector('a[href*="/scores/nba-basketball/matchup/"]');
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasMatchup && hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const rows = [];
      for (const eidNode of Array.from(tbody.querySelectorAll("[data-horizontal-eid]"))) {
        const eventId = eidNode.getAttribute("data-horizontal-eid");
        const root = ascendToGameRoot(eidNode);
        const matchupA = root.querySelector('a[href*="/scores/nba-basketball/matchup/"]');
        if (!eventId || !matchupA) continue;

        rows.push({
          event_id: eventId,
          matchup_url: matchupA.getAttribute("href"),
        });
      }
      return rows;
    }
    """

    payload: list[dict[str, Any]] = await page.evaluate(js)
    matchups: list[GameMatchup] = []
    seen: set[str] = set()
    for row in payload:
        event_id = str(row["event_id"])
        if event_id in seen:
            continue
        seen.add(event_id)
        matchup_url = _absolute_sbr_url(str(row["matchup_url"]))
        matchups.append(
            GameMatchup(
                event_id=event_id,
                game_date=page_date,
                season_year=season_year_for_date(page_date),
                start_time=None,
                matchup_url=matchup_url,
                line_history_url=build_sbr_line_history_url(event_id),
                team_away=None,
                team_home=None,
            )
        )

    return matchups


async def extract_matchup_records_from_page(
    page: Page,
    game: GameMatchup,
) -> pd.DataFrame:
    await page.goto(game.matchup_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_line_history_consent(page)

    try:
        await page.wait_for_selector(
            'section [class*="Rows_Row"]',
            timeout=TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        return pd.DataFrame()

    raw_sections: list[dict[str, Any]] = await page.evaluate(
        r"""
        () => {
          const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
          const normNode = (node) => norm((node && (node.innerText || node.textContent)) || "");
          const sections = [];

          for (const heading of Array.from(document.querySelectorAll("section h2"))) {
            const block = heading.parentElement;
            if (!block) continue;

            const rows = Array.from(block.querySelectorAll('[class*="Rows_Row"]'))
              .map(row => {
                const label = normNode(row.querySelector('[class*="Rows_RowLabel"]'));
                const values = Array.from(row.querySelectorAll('[class*="Rows_RowData"]'))
                  .map(cell => normNode(cell))
                  .filter(Boolean);
                return { label, values };
              })
              .filter(row => row.values.length >= 2);

            if (!rows.length) continue;
            sections.push({
              section: normNode(heading),
              rows,
            });
          }

          return sections;
        }
        """
    )

    matchup_record: dict[str, Any] = {
        "game_date": game.game_date,
        "season_year": game.season_year,
        "event_id": game.event_id,
        "start_time": game.start_time,
        "matchup_url": game.matchup_url,
        "team_away": game.team_away,
        "team_home": game.team_home,
    }
    for section in raw_sections:
        section_name = str(section.get("section") or "")
        section_slug = _slugify_matchup_key(section_name)
        if not section_slug:
            continue

        rows = section.get("rows") or []
        for row in rows:
            values = list(row.get("values") or [])
            if len(values) < 2:
                continue

            away_value = str(values[0]) if values[0] is not None else None
            home_value = str(values[1]) if values[1] is not None else None
            label = str(row.get("label") or "")
            label_slug = _slugify_matchup_key(label)
            if not label_slug:
                matchup_record["team_away"] = matchup_record["team_away"] or away_value
                matchup_record["team_home"] = matchup_record["team_home"] or home_value
                continue

            base_col = f"matchup_{section_slug}_{label_slug}"
            matchup_record[f"{base_col}_away"] = away_value
            matchup_record[f"{base_col}_home"] = home_value

    if len(matchup_record) == 7:
        return pd.DataFrame()

    return pd.DataFrame([matchup_record])


async def discover_line_history_bookmakers(page: Page) -> list[str]:
    try:
        await page.wait_for_selector(
            '#LineHistory aside [class*="GameMatchup_dropDownContainer"] img[alt$="Logo"]',
            timeout=TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        return []

    js = r"""
    () => {
      const items = Array.from(
        document.querySelectorAll('#LineHistory aside [class*="GameMatchup_dropDownContainer"] .dropdown-menu li')
      );
      const books = [];
      for (const item of items) {
        const img = Array.from(item.querySelectorAll("img[alt]"))
          .find(img => (img.getAttribute("alt") || "").endsWith("Logo"));
        if (img) books.push(img.getAttribute("alt"));
      }
      return [...new Set(books)];
    }
    """
    return await page.evaluate(js)


async def get_selected_bookmaker(page: Page) -> str | None:
    js = r"""
    () => {
      const button = document.querySelector('#LineHistory aside [class*="GameMatchup_dropDownContainer"] .dropdown > button');
      if (!button) return null;
      const img = Array.from(button.querySelectorAll("img[alt]"))
        .find(img => (img.getAttribute("alt") || "").endsWith("Logo"));
      return img ? img.getAttribute("alt") : null;
    }
    """
    return await page.evaluate(js)


async def select_bookmaker(page: Page, bookmaker_logo_alt: str) -> None:
    selected = await get_selected_bookmaker(page)
    if selected == bookmaker_logo_alt:
        return

    button = page.locator(
        '#LineHistory aside [class*="GameMatchup_dropDownContainer"] .dropdown > button'
    ).first
    await button.click(timeout=TIMEOUT_MS)

    item = (
        page.locator(
            '#LineHistory aside [class*="GameMatchup_dropDownContainer"] .dropdown-menu li'
        )
        .filter(has=page.locator(f'img[alt="{bookmaker_logo_alt}"]'))
        .first
    )
    await item.click(timeout=TIMEOUT_MS)

    await page.wait_for_function(
        """
        (bookmaker) => {
          const button = document.querySelector('#LineHistory aside [class*="GameMatchup_dropDownContainer"] .dropdown > button');
          if (!button) return false;
          return Array.from(button.querySelectorAll("img[alt]"))
            .some(img => img.getAttribute("alt") === bookmaker);
        }
        """,
        arg=bookmaker_logo_alt,
        timeout=TIMEOUT_MS,
    )


async def select_market(page: Page, market: str) -> None:
    data_format = MARKET_FORMATS[market]
    item = page.locator(f'#LineHistory .col-lg-9 li[data-format="{data_format}"]').first
    await item.click(timeout=TIMEOUT_MS)
    await page.wait_for_function(
        """
        (dataFormat) => {
          const items = Array.from(document.querySelectorAll('#LineHistory .col-lg-9 li[data-format]'));
          return items.some((item) => {
            const rect = item.getBoundingClientRect();
            return item.getAttribute("data-format") === dataFormat
              && item.className.includes("Dropdown_active")
              && rect.width > 0
              && rect.height > 0;
          });
        }
        """,
        arg=data_format,
        timeout=TIMEOUT_MS,
    )


async def extract_current_line_history_rows(
    page: Page,
    game: GameMatchup,
    *,
    bookmaker_logo_alt: str,
    market: str,
) -> list[dict[str, Any]]:
    raw_rows: list[list[str]] = await page.evaluate(
        r"""
        () => {
          const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
          return Array.from(document.querySelectorAll('#LineHistory aside [class*="Rows_Row"]'))
            .map(row => Array.from(row.querySelectorAll('[class*="Rows_RowData"]')).map(cell => norm(cell.textContent)))
            .filter(row => row.length >= 3);
        }
        """
    )

    bookmaker = bookmaker_logo_alt.replace(" Logo", "").strip()
    bookmaker_slug = _slugify_bookmaker(bookmaker_logo_alt)
    out: list[dict[str, Any]] = []
    section = "history"
    labels: tuple[str | None, str | None] = (None, None)
    change_order = 0

    for raw_row in raw_rows:
        row = (raw_row + ["", "", ""])[:3]
        row0, row1, row2 = row

        if row1.lower() == "opener":
            section = "opener"
            continue

        if row0.lower() == "time":
            labels = (row1 or None, row2 or None)
            continue

        if not row0:
            continue

        left_line, left_price = _parse_odds_value(row1, market)
        right_line, right_price = _parse_odds_value(row2, market)
        parsed_time = _parse_history_datetime(row0, game.game_date)
        row_kind = section

        out.append(
            {
                "game_date": game.game_date,
                "season_year": game.season_year,
                "event_id": game.event_id,
                "start_time": game.start_time,
                "matchup_url": game.matchup_url,
                "line_history_url": game.line_history_url,
                "team_away": game.team_away,
                "team_home": game.team_home,
                "bookmaker": bookmaker,
                "bookmaker_slug": bookmaker_slug,
                "market": market,
                "row_kind": row_kind,
                "change_order": change_order,
                "timestamp_raw": row0,
                "timestamp": parsed_time,
                "left_label": labels[0],
                "right_label": labels[1],
                "left_value_raw": row1 or None,
                "right_value_raw": row2 or None,
                "left_line": left_line,
                "left_price": left_price,
                "right_line": right_line,
                "right_price": right_price,
            }
        )
        change_order += 1

        if section == "opener":
            section = "history"

    return out


async def scrape_game_line_history(
    page: Page,
    game: GameMatchup,
    *,
    markets: list[str] | None = None,
    bookmaker_logo_alts: list[str] | None = None,
    include_matchup_records: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = markets or list(MARKET_FORMATS)
    matchup_records_df = pd.DataFrame()

    if include_matchup_records:
        try:
            matchup_records_df = await extract_matchup_records_from_page(page, game)
            if not matchup_records_df.empty:
                matchup_record = matchup_records_df.iloc[0]
                game = replace(
                    game,
                    team_away=matchup_record.get("team_away") or game.team_away,
                    team_home=matchup_record.get("team_home") or game.team_home,
                )
            await random_sleep(0.15, 0.35)
        except Exception as exc:
            print(
                f"Failed matchup records scrape for {game.game_date.isoformat()} "
                f"event_id={game.event_id}: {exc}"
            )

    await page.goto(
        game.line_history_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS
    )
    await try_click_line_history_consent(page)

    try:
        await page.wait_for_selector("#LineHistory aside", timeout=TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return pd.DataFrame(), matchup_records_df

    if bookmaker_logo_alts is None:
        bookmaker_logo_alts = await discover_line_history_bookmakers(page)

    rows: list[dict[str, Any]] = []
    for bookmaker_logo_alt in bookmaker_logo_alts:
        await select_bookmaker(page, bookmaker_logo_alt)
        await random_sleep(0.05, 0.15)

        for market in markets:
            await select_market(page, market)
            await random_sleep(0.05, 0.15)
            rows.extend(
                await extract_current_line_history_rows(
                    page,
                    game,
                    bookmaker_logo_alt=bookmaker_logo_alt,
                    market=market,
                )
            )

    return pd.DataFrame(rows), matchup_records_df


def _coerce_line_history_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "change_order",
        "left_line",
        "left_price",
        "right_line",
        "right_price",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


async def scrape_sportsbook_line_history_days(
    days: list[date],
    *,
    headless: bool = True,
    landing_market: str = "totals",
    markets: list[str] | None = None,
    include_matchup_records: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not days:
        return pd.DataFrame(), pd.DataFrame()

    line_history_frames: list[pd.DataFrame] = []
    matchup_record_frames: list[pd.DataFrame] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 1600})
        await context.route("**/*", block_heavy_assets)
        page = await context.new_page()

        for day in days:
            matchups = await extract_matchups_from_odds_page(
                page, day, source_market=landing_market
            )

            for game in matchups:
                try:
                    (
                        line_history_df,
                        matchup_records_df,
                    ) = await scrape_game_line_history(
                        page,
                        game,
                        markets=markets,
                        include_matchup_records=include_matchup_records,
                    )
                    if not line_history_df.empty:
                        line_history_frames.append(line_history_df)
                    if not matchup_records_df.empty:
                        matchup_record_frames.append(matchup_records_df)
                except Exception as exc:
                    print(
                        f"Failed line-history scrape for {day.isoformat()} "
                        f"event_id={game.event_id}: {exc}"
                    )
                await random_sleep(SLEEP_MIN_S, SLEEP_MAX_S)

        await browser.close()

    if line_history_frames:
        line_history_out = pd.concat(line_history_frames, ignore_index=True)
    else:
        line_history_out = pd.DataFrame()

    line_history_out = _coerce_line_history_numeric(line_history_out)

    if matchup_record_frames:
        matchup_records_out = pd.concat(matchup_record_frames, ignore_index=True)
    else:
        matchup_records_out = pd.DataFrame()

    return line_history_out, matchup_records_out


async def download_sportsbook_line_history_season_daily(
    season_year: int,
    *,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    headless: bool = True,
    landing_market: str = "totals",
    markets: list[str] | None = None,
    include_matchup_records: bool = True,
    overwrite_existing: bool = False,
) -> None:
    days = _season_date_range(
        season_year,
        start_month=start_month,
        start_day=start_day,
        end_month=end_month,
        end_day=end_day,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    remaining_days: list[date] = []
    skipped_days = 0
    for day in days:
        paths = _daily_output_paths(output_root, season_year, day)
        if (
            not overwrite_existing
            and paths.line_history_csv.exists()
            and paths.matchup_records_csv.exists()
        ):
            skipped_days += 1
            continue
        remaining_days.append(day)

    print(
        f"Season {_season_label(season_year)} | total_days={len(days)} | "
        f"already_scraped={skipped_days} | days_left_to_scrape={len(remaining_days)}"
    )
    if not remaining_days:
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 1600})
        await context.route("**/*", block_heavy_assets)
        page = await context.new_page()

        progress = tqdm(remaining_days, desc="Scraping SBR line history", unit="day")
        for day in progress:
            progress.set_postfix_str(day.isoformat())
            paths = _daily_output_paths(output_root, season_year, day)

            line_history_frames: list[pd.DataFrame] = []
            matchup_record_frames: list[pd.DataFrame] = []

            try:
                matchups = await extract_matchups_from_odds_page(
                    page, day, source_market=landing_market
                )

                for game in matchups:
                    try:
                        (
                            line_history_df,
                            matchup_records_df,
                        ) = await scrape_game_line_history(
                            page,
                            game,
                            markets=markets,
                            include_matchup_records=include_matchup_records,
                        )
                        if not line_history_df.empty:
                            line_history_frames.append(line_history_df)
                        if not matchup_records_df.empty:
                            matchup_record_frames.append(matchup_records_df)
                    except Exception as exc:
                        with paths.errors_log.open("a", encoding="utf-8") as f:
                            f.write(
                                f"{datetime.now().isoformat()} | {day.isoformat()} "
                                f"| event_id={game.event_id} | {repr(exc)}\n"
                            )
                    await random_sleep(SLEEP_MIN_S, SLEEP_MAX_S)

                if line_history_frames:
                    line_history_day_df = pd.concat(
                        line_history_frames, ignore_index=True
                    )
                else:
                    line_history_day_df = pd.DataFrame()
                line_history_day_df = _coerce_line_history_numeric(line_history_day_df)

                if matchup_record_frames:
                    matchup_records_day_df = pd.concat(
                        matchup_record_frames, ignore_index=True
                    )
                else:
                    matchup_records_day_df = pd.DataFrame()

                _with_columns(line_history_day_df, LINE_HISTORY_COLUMNS).to_csv(
                    paths.line_history_csv, index=False
                )
                _with_columns(matchup_records_day_df, MATCHUP_RECORD_COLUMNS).to_csv(
                    paths.matchup_records_csv, index=False
                )
                tqdm.write(
                    f"Wrote {day.isoformat()} | "
                    f"line_history_rows={len(line_history_day_df)} | "
                    f"matchup_record_rows={len(matchup_records_day_df)}"
                )
            except Exception as exc:
                with paths.errors_log.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now().isoformat()} | {day.isoformat()} | {repr(exc)}\n"
                    )

            await random_sleep(SLEEP_MIN_S, SLEEP_MAX_S)

        await browser.close()


def _parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def _date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _season_date_range(
    season_year: int,
    *,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> list[date]:
    start = date(season_year, start_month, start_day)
    end_year = (
        season_year
        if (end_month, end_day) >= (start_month, start_day)
        else season_year + 1
    )
    end = date(end_year, end_month, end_day)
    return _date_range(start, end)


def _season_label(season_year: int) -> str:
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def _daily_output_paths(
    output_root: Path,
    season_year: int,
    day: date,
) -> DailyOutputPaths:
    season_root = output_root / _season_label(season_year)
    line_history_dir = season_root / "line_history"
    matchup_records_dir = season_root / "matchup_records"
    line_history_dir.mkdir(parents=True, exist_ok=True)
    matchup_records_dir.mkdir(parents=True, exist_ok=True)

    return DailyOutputPaths(
        season_root=season_root,
        line_history_dir=line_history_dir,
        matchup_records_dir=matchup_records_dir,
        line_history_csv=line_history_dir / f"{day.isoformat()}_line_history.csv",
        matchup_records_csv=matchup_records_dir
        / f"{day.isoformat()}_matchup_records.csv",
        errors_log=season_root / "errors.log",
    )


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df


def _parse_markets(values: list[str] | None) -> list[str] | None:
    if not values:
        return None

    markets: list[str] = []
    for value in values:
        normalized = value.strip().lower().replace("-", "_")
        if normalized not in MARKET_FORMATS:
            raise ValueError(
                f"Unknown market {value!r}. Valid values: {sorted(MARKET_FORMATS)}"
            )
        markets.append(normalized)
    return markets


async def _scrape_event_id(
    event_id: str,
    game_date: date,
    *,
    headless: bool,
    markets: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    game = GameMatchup(
        event_id=event_id,
        game_date=game_date,
        season_year=season_year_for_date(game_date),
        start_time=None,
        matchup_url=f"{BASE_URL}/scores/nba-basketball/matchup/{event_id}/",
        line_history_url=build_sbr_line_history_url(event_id),
        team_away=None,
        team_home=None,
    )
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 1600})
        await context.route("**/*", block_heavy_assets)
        page = await context.new_page()
        line_history_df, matchup_records_df = await scrape_game_line_history(
            page, game, markets=markets
        )
        await browser.close()
    return line_history_df, matchup_records_df


if __name__ == "__main__":

    season_year = 2020
    season_start_month = 7 #6
    season_start_day = 29 #30
    season_end_month = 11 #11
    season_end_day = 30 #31
    headless = True
    # Landing page used only to find matchup links. It does not limit line-history markets.
    landing_market = "totals"
    # None scrapes all line-history markets: totals, money line, and point spread.
    # To limit line-history tabs, use e.g. ["totals"].
    markets = None
    include_matchup_records = True
    output_root = DEFAULT_OUTPUT_ROOT
    # False skips days where both daily CSVs already exist. True re-scrapes them.
    overwrite_existing = False

    asyncio.run(
        download_sportsbook_line_history_season_daily(
            season_year,
            start_month=season_start_month,
            start_day=season_start_day,
            end_month=season_end_month,
            end_day=season_end_day,
            output_root=output_root,
            headless=headless,
            landing_market=landing_market,
            markets=markets,
            include_matchup_records=include_matchup_records,
            overwrite_existing=overwrite_existing,
        )
    )
