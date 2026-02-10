#!/usr/bin/env python3
"""
Create training dataset up to 2026-01-10 (no date-to-predict / scheduled games).

This script calls `create_df_to_predict` without providing a prediction date
or scheduled-game data. It saves the resulting DataFrame to
`data/train_data/training_dataYYYYMMDD.csv`.
"""

import pandas as pd
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict


def main(limit_date_to_train: str = "2026-01-10") -> None:
    """Create training data up to `limit_date_to_train`.

    Args:
        limit_date_to_train: Date string YYYY-MM-DD (default: 2026-01-10)
    """

    # Call create_df_to_predict without a scheduled date (no todays prediction)
    df_train = create_df_to_predict(
        todays_prediction=False,
        scheduled_games=None,
        df_referees_scheduled=None,
        injury_dict_scheduled=None,
        df_odds_scheduled=None,
        recent_limit_to_include=limit_date_to_train,
        older_limit_to_include=None
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
        default="2026-01-20",
        help="Limit date to train (YYYY-MM-DD). Defaults to 2026-01-10",
    )

    args = parser.parse_args()
    main(args.limit)
