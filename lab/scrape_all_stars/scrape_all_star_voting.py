from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

# =========================
# CONFIG
# =========================
START_URL: str = (
    "https://www.basketball-reference.com/allstar/NBA_2026_voting.html"
)
START_YEAR: int = 2019
END_YEAR: int = 2026

OUT_DIR: Path = Path(
    "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/all_star_voting"
)
SAVE_CSV: bool = True
HEADLESS: bool = False
TIMEOUT_MS: int = 30_000
SLEEP_BETWEEN_PAGES_S: float = 1.5
# =========================


EXTRA_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

CONFERENCE_LABELS: dict[str, str] = {
    "eastern": "Eastern Conference",
    "western": "Western Conference",
}

POSITION_LABELS: dict[str, str] = {
    "backcourt": "Backcourt",
    "frontcourt": "Frontcourt",
    "all": "All Positions",
}

TABLE_STATS: tuple[str, ...] = (
    "season",
    "player_name",
    "fan_votes",
    "fan_rank",
    "player_votes",
    "player_rank",
    "media_votes",
    "media_rank",
    "score",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "fan_votes",
    "fan_rank",
    "player_votes",
    "player_rank",
    "media_votes",
    "media_rank",
    "score",
)


@dataclass(frozen=True)
class VotingPage:
    slug: str
    conference_slug: str
    conference: str
    position_slug: str
    position: str
    url: str


def find_voting_table(soup: BeautifulSoup):
    table = soup.select_one("table.stats_table")
    if table is None:
        table = soup.find("table", id=re.compile(r"^(bc|fc|x)-[we]$"))
    return table


def fetch_html(page: Page, url: str) -> str:
    response = page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    if response is None:
        raise RuntimeError(f"No response returned for URL: {url}")
    if response.status == 429:
        raise RuntimeError(f"Basketball Reference returned HTTP 429 for URL: {url}")
    if response.status >= 400:
        raise RuntimeError(
            f"Basketball Reference returned HTTP {response.status} for URL: {url}"
        )

    try:
        page.wait_for_selector("table.stats_table", timeout=TIMEOUT_MS)
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(f"No voting table loaded for URL: {url}") from exc

    return page.content()


def season_year_from_url(url: str) -> int:
    match = re.search(r"NBA_(\d{4})_voting", url)
    if not match:
        raise ValueError(f"Could not infer all-star voting year from URL: {url}")
    return int(match.group(1))


def build_voting_url(season_year: int) -> str:
    return f"https://www.basketball-reference.com/allstar/NBA_{season_year}_voting.html"


def discover_voting_pages(
    page: Page, start_url: str
) -> list[VotingPage]:
    html = fetch_html(page, start_url)
    soup = BeautifulSoup(html, "lxml")
    pages_by_slug: dict[str, VotingPage] = {}

    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        match = re.search(
            r"NBA_\d{4}_voting-(?:(backcourt|frontcourt)-)?"
            r"(eastern|western)-conference\.html",
            href,
        )
        if not match:
            continue

        position_slug = match.group(1) or "all"
        conference_slug = match.group(2)
        slug = f"{position_slug}-{conference_slug}"
        pages_by_slug[slug] = VotingPage(
            slug=slug,
            conference_slug=conference_slug,
            conference=CONFERENCE_LABELS[conference_slug],
            position_slug=position_slug,
            position=POSITION_LABELS[position_slug],
            url=urljoin(start_url, href),
        )

    if not pages_by_slug:
        table = find_voting_table(soup)
        if table is None:
            raise ValueError(f"No voting table or voting links found: {start_url}")
        conference_slug = conference_slug_from_table(table, start_url)
        position_slug = position_slug_from_table(table, start_url)
        slug = f"{position_slug}-{conference_slug}"
        pages_by_slug[slug] = VotingPage(
            slug=slug,
            conference_slug=conference_slug,
            conference=CONFERENCE_LABELS[conference_slug],
            position_slug=position_slug,
            position=POSITION_LABELS[position_slug],
            url=start_url,
        )

    sort_order = {
        "backcourt-western": 0,
        "frontcourt-western": 1,
        "backcourt-eastern": 2,
        "frontcourt-eastern": 3,
        "all-western": 4,
        "all-eastern": 5,
    }
    return sorted(
        pages_by_slug.values(),
        key=lambda page: sort_order.get(page.slug, len(sort_order)),
    )


def conference_slug_from_table(table, url: str) -> str:
    table_id = str(table.get("id", ""))
    caption = table.find("caption")
    caption_text = caption.get_text(" ", strip=True).lower() if caption else ""
    haystack = f"{table_id} {caption_text} {url}".lower()

    if "eastern" in haystack or "x-e" in haystack:
        return "eastern"
    if "western" in haystack or "x-w" in haystack:
        return "western"

    raise ValueError("Could not infer conference from voting table")


def position_slug_from_table(table, url: str) -> str:
    table_id = str(table.get("id", ""))
    caption = table.find("caption")
    caption_text = caption.get_text(" ", strip=True).lower() if caption else ""
    haystack = f"{table_id} {caption_text} {url}".lower()

    if "backcourt" in haystack or "bc-" in haystack:
        return "backcourt"
    if "frontcourt" in haystack or "fc-" in haystack:
        return "frontcourt"
    return "all"


def text_for_stat(row, stat: str) -> str:
    cell = row.find(["th", "td"], attrs={"data-stat": stat})
    if cell is None:
        return ""
    return cell.get_text(" ", strip=True)


def href_for_stat(row, stat: str) -> str:
    cell = row.find(["th", "td"], attrs={"data-stat": stat})
    if cell is None:
        return ""
    link = cell.find("a", href=True)
    return str(link["href"]) if link else ""


def parse_voting_table(
    html: str, page: VotingPage, season_year: int, scraped_at: str
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = find_voting_table(soup)
    if table is None:
        raise ValueError(f"No table found for {page.slug}: {page.url}")

    rows: list[dict[str, str]] = []
    for row in table.select("tbody tr"):
        if "thead" in (row.get("class") or []):
            continue

        parsed_row = {stat: text_for_stat(row, stat) for stat in TABLE_STATS}
        if not parsed_row["player_name"]:
            continue

        player_url = href_for_stat(row, "player_name")
        rows.append(
            {
                "all_star_year": season_year,
                "conference": page.conference,
                "position": page.position,
                **parsed_row,
                "player_url": urljoin(page.url, player_url) if player_url else "",
                "source_url": page.url,
                "scraped_at": scraped_at,
            }
        )

    if not rows:
        raise ValueError(f"No player rows found for {page.slug}: {page.url}")

    df = pd.DataFrame(rows)
    return clean_voting_df(df)


def clean_voting_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    return out[
        [
            "all_star_year",
            "conference",
            "position",
            "season",
            "player_name",
            "fan_votes",
            "fan_rank",
            "player_votes",
            "player_rank",
            "media_votes",
            "media_rank",
            "score",
            "player_url",
            "source_url",
            "scraped_at",
        ]
    ]


def output_dir_for_year(year: int) -> Path:
    out_dir = OUT_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_season_csv(page_dfs: Iterable[pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    dfs = list(page_dfs)
    combined_df = pd.concat(dfs, ignore_index=True)

    if SAVE_CSV:
        combined_path = out_dir / "all_conferences.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved {len(combined_df):,} rows to {combined_path}")

    return combined_df


def scrape_season(season_year: int, page: Page) -> pd.DataFrame:
    start_url = build_voting_url(season_year)
    out_dir = output_dir_for_year(season_year)
    scraped_at = datetime.now(UTC).isoformat()

    voting_pages = discover_voting_pages(page, start_url)
    names = ", ".join(f"{item.position} {item.conference}" for item in voting_pages)
    print(f"{season_year}: discovered {len(voting_pages)} voting page(s): {names}")

    page_dfs: list[pd.DataFrame] = []
    for voting_page in voting_pages:
        time.sleep(SLEEP_BETWEEN_PAGES_S)
        html = fetch_html(page, voting_page.url)
        df = parse_voting_table(html, voting_page, season_year, scraped_at)
        page_dfs.append(df)

    return save_season_csv(page_dfs, out_dir)


def scrape_seasons(start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    season_dfs: list[pd.DataFrame] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            extra_http_headers=EXTRA_HTTP_HEADERS,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()
        try:
            for season_year in range(start_year, end_year + 1):
                season_dfs.append(scrape_season(season_year, page=page))
                time.sleep(SLEEP_BETWEEN_PAGES_S)
        finally:
            context.close()
            browser.close()

    combined_df = pd.concat(season_dfs, ignore_index=True)
    if SAVE_CSV:
        combined_path = OUT_DIR / f"all_star_voting_{start_year}_{end_year}.csv"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved {len(combined_df):,} rows to {combined_path}")

    return combined_df


def run(start_url: str = START_URL) -> pd.DataFrame:
    season_year = season_year_from_url(start_url)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            extra_http_headers=EXTRA_HTTP_HEADERS,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()
        try:
            return scrape_season(season_year, page=page)
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    print("Starting all-star voting scrape...")
    print(f"Seasons: {START_YEAR} to {END_YEAR}")
    df_all = scrape_seasons(START_YEAR, END_YEAR)
    print(f"Scrape complete. Rows scraped: {len(df_all):,}")
