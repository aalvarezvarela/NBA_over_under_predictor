"""
NBA Over/Under Predictor - Main Execution Script

This is the main entry point for the NBA Over/Under prediction system.
It orchestrates the entire prediction pipeline:
1. Updates the game database with latest data
2. Fetches and processes injury reports
3. Prepares prediction features
4. Runs ML models for predictions
5. Exports results to Excel

Usage:
    From project root: python -m src.nba_predictor
    From src folder:   python nba_predictor.py
    With date:         python nba_predictor.py -d 2025-01-15
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import (
    create_df_to_predict,
)
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)

from models import predict_nba_games, save_predictions_to_excel
from scripts.update_databases.update_all_databases import update_all_databases


def print_banner(date_to_predict: str) -> None:
    line = "=" * 70
    title = "NBA OVER/UNDER PREDICTION SYSTEM"
    print(f"\n{line}")
    print(title.center(len(line)))
    print(line)
    print(f"  Prediction Date: {date_to_predict}")
    print(line + "\n")


def print_step_header(step_number: int, title: str) -> None:
    """Print a compact, eye-catching step header."""
    sep = "-" * 70
    header = f" STEP {step_number} — {title} "
    print(sep)
    print(header.center(len(sep)))
    print(sep)


def print_status(message: str, ok: bool = True) -> None:
    """Print a single-line status message with a check or cross."""
    symbol = "✓" if ok else "✖"
    print(f"  {symbol} {message}")


def main():
    """
    Main execution function for the NBA prediction pipeline.

    This function:
    1. Parses command-line arguments for the prediction date
    2. Loads configuration settings
    3. Updates the game database
    4. Processes data with injuries
    5. Generates predictions
    6. Saves results to Excel
    """

    date_to_predict = datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")

    # Print welcome banner
    print_banner(date_to_predict)
    print(
        f"  Models: regressor={SETTINGS.regressor_model_path} | classifier={SETTINGS.classifier_model_path}\n"
    )

    # Step 1: Update the database
    print_step_header(1, "Updating All Databases")
    season_to_update = str(date_to_predict)[:4]
    try:
        update_all_databases(
            start_season_year=int(season_to_update),
            end_season_year=int(season_to_update),
        )
        print_status("Databases updated")
    except Exception as e:
        print_status(f"Failed to update databases: {e}", ok=False)
        raise

    # Step 2: Fetch scheduled games, referees, injuries and odds
    print_step_header(2, "Fetching Scheduled Games & Reports")
    try:
        (
            scheduled_games,
            df_referees_scheduled,
            injury_dict_scheduled,
            df_odds_scheduled,
        ) = get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            save_reports_path=SETTINGS.report_path,
            odds_api_key=SETTINGS.odds_api_key,
            odds_base_url=SETTINGS.odds_base_url,
        )
        print_status("Fetched scheduled games, refs, injuries and odds")
    except Exception as e:
        print_status(f"Failed to fetch scheduled data: {e}", ok=False)
        raise

    # Step 3: Build feature DataFrame for prediction
    print_step_header(3, "Preparing Feature DataFrame")
    try:
        df_to_predict = create_df_to_predict(
            todays_prediction=True,
            scheduled_games=scheduled_games,
            df_referees_scheduled=df_referees_scheduled,
            injury_dict_scheduled=injury_dict_scheduled,
            df_odds_scheduled=df_odds_scheduled,
            recent_limit_to_include=date_to_predict,
        )
        print_status("Feature DataFrame prepared")
    except Exception as e:
        print_status(f"Failed to prepare features: {e}", ok=False)
        raise

    if df_to_predict.empty:
        print("⚠️  Warning: No games found for the specified date.")
        print("    Please check if there are scheduled games on this date.")
        return 0

    print_status(f"Found {len(df_to_predict)} game(s) to predict")

    # # Step 4: Generate predictions
    # print_step_header(4, "Generating Predictions")
    # try:
    #     predictions_dfs = predict_nba_games(df_to_predict)
    #     print_status("Predictions generated")
    # except Exception as e:
    #     print_status(f"Failed to generate predictions: {e}", ok=False)
    #     raise


if __name__ == "__main__":
    main()
