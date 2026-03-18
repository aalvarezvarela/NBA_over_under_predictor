#!/usr/bin/env python3
"""
Create training dataset up to 2026-01-10 (no date-to-predict / scheduled games).

This script calls `create_df_to_predict` without providing a prediction date
or scheduled-game data. It saves the resulting DataFrame to
`data/train_data/training_dataYYYYMMDD.csv`.
"""

import pandas as pd
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict


def main(
    limit_date_to_train: str = "2026-01-10", n_seasons_to_include: int = None
) -> None:
    """Create training data up to `limit_date_to_train`.

    Args:
        limit_date_to_train: Date string YYYY-MM-DD (default: 2026-01-10)
        n_seasons_to_include: Number of seasons to include (default: None, uses all from 2017-18)
    """

    # Call create_df_to_predict without a scheduled date (no todays prediction)
    df_train = create_df_to_predict(
        todays_prediction=False,
        recent_limit_to_include=limit_date_to_train,
        older_season_limit=n_seasons_to_include,
    )

    output_path = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data"
    )
    output_name = f"{output_path}/all_odds_training_data_until_{pd.to_datetime(limit_date_to_train).strftime('%Y%m%d')}.csv"

    # Save to CSV
    df_train.to_csv(output_name, index=False)
    print(f"Training data saved to {output_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create training dataset up to a given date (default: 2026-01-10)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        dest="limit",
        default="2026-03-04",
        help="Limit date to train (YYYY-MM-DD). Defaults to 2026-03-04",
    )
    parser.add_argument(
        "--n-seasons",
        "-n",
        dest="n_seasons",
        type=int,
        default=None,
        help="Number of seasons to include. Defaults to None (all from 2017-18)",
    )

    args = parser.parse_args()
    main(args.limit, args.n_seasons)
