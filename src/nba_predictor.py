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

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent directory to path if running from src folder
if __name__ == "__main__":
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    # If we're in the src directory, add it to sys.path
    if current_dir.name == "src" and str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from config import LEGEND, settings
from data_processing import create_df_to_predict
from fetch_data.manage_games_database import update_database
from fetch_data.manage_odds_data.update_odds import update_odds
from models import predict_nba_games, save_predictions_to_excel


def print_step_header(step_number: int, title: str) -> None:
    """
    Prints a formatted header for each pipeline step.

    Args:
        step_number: The step number in the pipeline
        title: Description of the step
    """
    print("\n" + "=" * 60)
    print(f"STEP {step_number}: {title}")
    print("=" * 60 + "\n")


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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="NBA Over/Under Prediction System - Generate predictions for NBA games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Generate predictions for today's games
  %(prog)s -d 2025-01-15     Generate predictions for January 15, 2025
        """,
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        help="Prediction date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "-s",
        "--save-excel",
        action="store_true",
        default=False,
        help="If set, save predictions to Excel. Default: False",
    )

    args = parser.parse_args()

    # Determine the date to predict (using Pacific Time - US West Coast)
    pacific_tz = ZoneInfo("America/Los_Angeles")

    if args.date:
        # Handle special keywords
        if args.date.lower() == "tomorrow":
            date_to_predict = (datetime.now(pacific_tz) + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
        elif args.date.lower() == "today":
            date_to_predict = datetime.now(pacific_tz).strftime("%Y-%m-%d")
        else:
            # Try to parse as a date string
            try:
                date_to_predict = datetime.strptime(args.date, "%Y-%m-%d").strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                print(
                    "❌ Error: Invalid date format. Please use YYYY-MM-DD, 'today', or 'tomorrow'."
                )
                return 1
    else:
        date_to_predict = datetime.now(pacific_tz).strftime("%Y-%m-%d")

    # Print welcome message
    print("\n" + "=" * 60)
    print("  NBA OVER/UNDER PREDICTION SYSTEM")
    print("=" * 60)
    print(f"📅 Prediction Date: {date_to_predict}")
    print(f"📁 Data Folder: {settings.data_folder}")
    print(f"💰 Odds Data Folder: {settings.odds_data_folder}")
    print(f"🤖 Regressor Model: {settings.regressor_model_path}")
    print(f"🤖 Classifier Model: {settings.classifier_model_path}")
    print(f"📊 Output Folder: {settings.output_path}")

    data_folder = settings.get_absolute_path(settings.data_folder)

    try:
        # Step 1: Update the database
        print_step_header(1, "Updating Game Database")
        limit = True
        while limit:
            limit = update_database(
                str(data_folder / "season_games_data/"),
                date=datetime.strptime(date_to_predict, "%Y-%m-%d"),
                save_csv=args.save_excel,
            )

        # Step 2: Update odds data
        print_step_header(2, "Fetching Betting Odds Data")
        odds_folder = settings.get_absolute_path(settings.odds_data_folder) or None
        if args.save_excel and odds_folder is not None:
            odds_folder.mkdir(parents=True, exist_ok=True)

        df_odds = update_odds(
            date_to_predict=date_to_predict,
            odds_folder=str(odds_folder),
            ODDS_API_KEY=settings.odds_api_key,
            BASE_URL=settings.odds_base_url,
            save_csv=args.save_excel,
        )
        print("✓ Odds data updated successfully")

        # Step 3: Prepare data for prediction
        print_step_header(3, "Processing Data and Injury Reports")
        df_to_predict = create_df_to_predict(
            data_path=str(data_folder),
            date_to_predict=date_to_predict,
            nba_injury_reports_url=settings.nba_injury_reports_url,
            df_odds=df_odds,
            reports_path=str(settings.get_absolute_path(settings.report_path)),
            filter_for_date_to_predict=True,
        )

        if df_to_predict.empty:
            print("⚠️  Warning: No games found for the specified date.")
            print("    Please check if there are scheduled games on this date.")
            return 0

        print(f"✓ Found {len(df_to_predict)} games to predict")

        # Step 4: Generate predictions
        print_step_header(4, "Generating Predictions")
        predictions_dfs = predict_nba_games(df_to_predict)

        # Step 6: Save predictions (optional)
        if args.save_excel:
            print_step_header(5, "Saving Results")
            output_dir = settings.get_absolute_path(settings.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"NBA_predictions_{date_to_predict}.xlsx"
            save_predictions_to_excel(predictions_dfs, str(output_file), LEGEND)
            print("\n" + "=" * 60)
            print("  ✓ PREDICTION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📄 Results saved to: {output_file}")
            print(f"📊 Total games analyzed: {len(df_to_predict)}")
            print("\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found - {e}")
        print("   Please check your configuration and data files.")
        raise e
    except Exception as e:
        print(f"\n❌ Error during prediction pipeline: {e}")
        import traceback

        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()
