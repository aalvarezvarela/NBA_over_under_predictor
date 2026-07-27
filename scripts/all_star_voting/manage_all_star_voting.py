from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.postgre_db.all_star_voting.process_all_star_voting_data import (
    DEFAULT_INPUT_CSV,
    PROJECT_ROOT,
    prepare_all_star_voting_dataset,
)
from nba_ou.postgre_db.all_star_voting.upload_data_all_star_voting import (
    build_and_upload_all_star_voting,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data/all_star_voting"
DEFAULT_AVAILABLE_CSV = DEFAULT_INPUT_CSV


def _season_csv_path(data_dir: Path, all_star_year: int) -> Path:
    return data_dir / str(all_star_year) / "all_conferences.csv"


def find_available_season_csvs(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[Path]:
    if not data_dir.exists():
        return []

    paths: list[Path] = []
    for season_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        try:
            all_star_year = int(season_dir.name)
        except ValueError:
            continue
        if start_year is not None and all_star_year < start_year:
            continue
        if end_year is not None and all_star_year > end_year:
            continue

        csv_path = season_dir / "all_conferences.csv"
        if csv_path.exists():
            paths.append(csv_path)

    return paths


def build_available_input_csv(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_csv: Path = DEFAULT_AVAILABLE_CSV,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    overwrite: bool = False,
) -> Path:
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{output_csv} already exists. Pass --overwrite-output to replace it."
        )

    csv_paths = find_available_season_csvs(
        data_dir, start_year=start_year, end_year=end_year
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No season all_conferences.csv files found under {data_dir}"
        )

    dfs = [pd.read_csv(path) for path in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(
        f"Wrote {len(combined):,} rows from {len(csv_paths)} season CSVs to {output_csv}"
    )
    return output_csv


def scrape_season(
    all_star_year: int,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    headless: bool = True,
    overwrite: bool = False,
) -> Path:
    output_csv = _season_csv_path(data_dir, all_star_year)
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{output_csv} already exists. Pass --overwrite to scrape it again."
        )

    from nba_ou.postgre_db.all_star_voting import scrape_all_star_voting as scraper

    scraper.OUT_DIR = data_dir
    scraper.HEADLESS = headless
    scraper.SAVE_CSV = True

    with scraper.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(
            extra_http_headers=scraper.EXTRA_HTTP_HEADERS,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()
        try:
            scraper.scrape_season(all_star_year, page=page)
        finally:
            context.close()
            browser.close()

    if not output_csv.exists():
        raise RuntimeError(
            f"Scrape completed but expected output was not found: {output_csv}"
        )
    return output_csv


def prepare_from_available_data(
    input_csv: Path | None,
    *,
    data_dir: Path,
    output_csv: Path,
    start_year: int | None,
    end_year: int | None,
    overwrite_output: bool,
) -> Path:
    if input_csv is not None:
        return input_csv
    if output_csv.exists() and not overwrite_output:
        print(
            f"Using existing combined input CSV {output_csv}. "
            "Pass --overwrite-output to rebuild it."
        )
        return output_csv
    return build_available_input_csv(
        data_dir=data_dir,
        output_csv=output_csv,
        start_year=start_year,
        end_year=end_year,
        overwrite=overwrite_output,
    )


def add_scrape_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "scrape-season",
        help="Scrape one Basketball Reference all-star voting season.",
    )
    parser.add_argument(
        "all_star_year",
        type=int,
        help="All-star game year, e.g. 2027 for NBA_2027_voting.html.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace data/all_star_voting/<year>/all_conferences.csv if it exists.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser headed instead of the default headless mode.",
    )
    parser.set_defaults(func=handle_scrape)


def add_build_input_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "build-input",
        help="Combine available per-season scrape CSVs into one input CSV.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_AVAILABLE_CSV)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.set_defaults(func=handle_build_input)


def add_upload_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "upload",
        help="Prepare and upsert all-star voting data into Supabase.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Use a specific combined input CSV. If omitted, available season CSVs are combined first.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--combined-output-csv", type=Path, default=DEFAULT_AVAILABLE_CSV
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite the generated combined input CSV.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate the Supabase all-star voting table before upload.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and prepare the dataset, but do not upload.",
    )
    parser.add_argument(
        "--skip-unresolved",
        action="store_true",
        help=(
            "Skip voting rows whose player_id or team cannot be resolved "
            "instead of aborting."
        ),
    )
    parser.set_defaults(func=handle_upload)


def handle_scrape(args: argparse.Namespace) -> None:
    output_csv = scrape_season(
        args.all_star_year,
        data_dir=args.data_dir,
        headless=not args.headed,
        overwrite=args.overwrite,
    )
    print(f"Scraped {args.all_star_year} to {output_csv}")


def handle_build_input(args: argparse.Namespace) -> None:
    build_available_input_csv(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite_output,
    )


def handle_upload(args: argparse.Namespace) -> None:
    input_csv = prepare_from_available_data(
        args.input_csv,
        data_dir=args.data_dir,
        output_csv=args.combined_output_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite_output=args.overwrite_output,
    )

    if args.prepare_only:
        df = prepare_all_star_voting_dataset(
            input_csv=input_csv, skip_unresolved=args.skip_unresolved
        )
        print(
            f"Prepared {len(df):,} rows for season_years "
            f"{sorted(df['season_year'].unique().tolist())}; upload skipped."
        )
        return

    result = build_and_upload_all_star_voting(
        input_csv=input_csv,
        drop_existing=args.drop_existing,
        skip_unresolved=args.skip_unresolved,
    )
    print(result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain all-star voting scrape files and Supabase table."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_scrape_parser(subparsers)
    add_build_input_parser(subparsers)
    add_upload_parser(subparsers)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
