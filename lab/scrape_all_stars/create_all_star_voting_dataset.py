from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.postgre_db.all_star_voting.process_all_star_voting_data import (
    DEFAULT_INPUT_CSV,
    PROJECT_ROOT,
    prepare_all_star_voting_dataset,
)

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "data/all_star_voting/all_star_voting_2019_2026_supabase_ready.csv"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Supabase-ready all-star voting CSV with player IDs."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = prepare_all_star_voting_dataset(input_csv=args.input_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
